import os
import json
import asyncio
import logging
import numpy as np
import requests
import torch
import re
from typing import Dict, List, Optional, Tuple
from pymongo import MongoClient
from dotenv import load_dotenv
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pprint import pprint

# 严格遵循官方MCP示例的导入方式
from mcp.server.fastmcp import FastMCP

# -------------------------- 基础配置 --------------------------
# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)
logger = logging.getLogger("enhanced-bio-invasion-mcp")

# 加载环境变量
load_dotenv()
ENV = os.getenv("MCP_ENV", "development")

# -------------------------- 全局资源初始化 --------------------------
# MongoDB连接
mongo_client = None
mongo_col = None

def init_mongo():
    """初始化MongoDB连接"""
    global mongo_client, mongo_col
    mongo_uri = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
    default_db = os.getenv('MONGO_DB_NAME', '中国生物入侵研究')
    default_collection = os.getenv('MONGO_COLLECTION', '生物入侵研究')
    
    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")  # 验证连接
        db = client[default_db]
        col = db[default_collection]
        logger.info(f"✅ MongoDB连接成功：{default_db}.{default_collection}")
        mongo_client = client
        mongo_col = col
        return True
    except Exception as e:
        logger.error(f"❌ MongoDB初始化失败：{str(e)}")
        return False

# 重排序模型
reranker_tokenizer = None
reranker_model = None

def init_reranker():
    """加载重排序模型（简化版本，避免内存问题）"""
    global reranker_tokenizer, reranker_model
    model_name = "BAAI/bge-reranker-base"  # 使用更小的模型
    
    try:
        # 简化模型加载，减少内存使用
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float32,  # 使用float32减少兼容性问题
            device_map="cpu"  # 强制使用CPU避免GPU内存问题
        )
        model.eval()
        
        reranker_tokenizer = tokenizer
        reranker_model = model
        logger.info(f"✅ 重排序模型加载成功：{model_name}")
        return True
    except Exception as e:
        logger.warning(f"⚠️  重排序模型加载失败：{str(e)}（将跳过重排序步骤）")
        return False

# 初始化资源
mongo_initialized = init_mongo()
reranker_initialized = init_reranker()

# -------------------------- 核心工具函数 --------------------------
def text_to_vector(text: str) -> np.ndarray:
    """使用Ollama嵌入模型将文本转换为向量"""
    try:
        import ollama
        
        # 文本归一化处理
        text = text.lower()  # 统一小写
        text = re.sub(r'[^\w\s]', '', text)  # 去除标点
        text = re.sub(r'\s+', ' ', text).strip()  # 标准化空格
        
        # 使用Ollama嵌入模型
        embedding_model = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:8b")
        
        try:
            # 使用主模型生成嵌入向量
            response = ollama.embeddings(model=embedding_model, prompt=text)
            vector = np.array(response["embedding"])
            logger.info(f"✅ 使用Ollama嵌入模型成功: {embedding_model}")
            return vector
        except Exception as e:
            logger.warning(f"⚠️  主嵌入模型 {embedding_model} 失败: {str(e)}，尝试备用模型")
            
            # 回退到nomic-embed-text模型
            try:
                response = ollama.embeddings(model="nomic-embed-text", prompt=text)
                vector = np.array(response["embedding"])
                logger.info(f"✅ 使用备用嵌入模型成功")
                return vector
            except Exception as fallback_error:
                raise RuntimeError(f"所有嵌入模型都失败: 主模型错误 - {str(e)}, 备用模型错误 - {str(fallback_error)}")
            
    except ImportError:
        logger.error("❌ ollama库未安装，请运行: pip install ollama")
        raise RuntimeError("ollama库未安装")
    except Exception as e:
        logger.error(f"❌ 文本转向量失败: {str(e)}")
        raise

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """计算余弦相似度"""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def rerank_results(query_text: str, results: List[Tuple[Dict, float]], tokenizer, model) -> List[Tuple[Dict, float]]:
    """使用reranker模型对结果进行重排序"""
    try:
        pairs = [(query_text, doc['metadata'].get('title', '') + " " + 
                 doc['metadata'].get('abstract', '')) for doc, _ in results]
        
        with torch.no_grad():
            # 减少最大长度以降低GPU内存使用
            inputs = tokenizer(
                pairs,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
                max_length=256,  # 从512减少到256
                pad_to_multiple_of=8
            )
            
            # 分批处理以避免内存溢出
            batch_size = 4  # 减少batch size
            scores = []
            for i in range(0, len(pairs), batch_size):
                batch_inputs = {k: v[i:i+batch_size].to(model.device) 
                              for k, v in inputs.items()}
                batch_scores = model(**batch_inputs, return_dict=True).logits.view(-1,).float()
                scores.append(batch_scores.cpu())
            
            scores = torch.cat(scores)
    except Exception as e:
        logger.error(f"重排序失败: {str(e)}")
        return results
        
    # 将分数转换为numpy数组并归一化到0-1范围
    scores = torch.sigmoid(scores).numpy()
    
    # 更新结果分数(提高reranker权重)
    reranked = [(doc, sim * 0.3 + score * 0.7)  # 30%原始相似度 + 70%reranker分数
               for (doc, sim), score in zip(results, scores)]
    
    # 按新分数排序
    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked

def enhance_results_with_llm(query_text: str, original_docs: List[Tuple[Dict, float, Optional[Dict]]]) -> Optional[str]:
    """使用DeepSeek API优化输出结果，结合原文档内容"""
    DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
    DEEPSEEK_API_URL = os.getenv('DEEPSEEK_API_URL')
    
    if not DEEPSEEK_API_KEY:
        logger.warning("⚠️  未配置DeepSeek API Key，跳过结果优化")
        return None
    
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # 准备包含原文档内容的详细结果摘要
    results_summary = "\n".join(
        f"结果{i} (相似度:{sim:.4f}):\n"
        f"- 分段编号: {original_doc['chunk_number'] if original_doc else '无'}\n"
        f"- 来源文件: {original_doc['source'] if original_doc else '未知'}\n"
        f"- 内容预览: {original_doc['content'][:200] + '...' if original_doc and len(original_doc['content']) > 200 else original_doc['content'] if original_doc else '无内容'}"
        for i, (doc, sim, original_doc) in enumerate(original_docs, 1)
    )
    
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "system",
                "content": "你是一个专业的生物入侵研究专家。请基于检索到的实际文档内容，生成关于查询主题的深度整合分析报告。报告要求：\n\n1. 关键信息提取：准确提取文档中的事实数据、时间、地点、影响范围等核心信息\n2. 内容深度分析：分析文档间的关联性、数据一致性、研究趋势\n3. 专业见解：提供基于文档证据的专业判断和风险评估\n4. 结构化输出：使用清晰的章节结构，包括摘要、分析、结论和建议\n5. 准确性：严格基于提供的文档内容，不添加外部知识或假设\n\n输出语言：中文\n报告风格：学术专业，数据驱动"
            },
            {
                "role": "user",
                "content": f"查询主题：{query_text}\n\n检索到的相关文档内容（按相似度排序）：\n{results_summary}\n\n请基于以上实际文档内容，生成一份专业的整合分析报告。要求：\n- 严格基于提供的文档内容进行分析\n- 提取关键数据和事实信息\n- 分析不同文档间的关联和一致性\n- 评估信息的完整性和可靠性\n- 提供专业的结论和建议\n- 使用清晰的章节结构组织内容"
            }
        ],
        "temperature": 0.2,
        "max_tokens": 1500
    }
    
    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"结果优化失败: {str(e)}")
        return None

# -------------------------- MCP服务器初始化 --------------------------
# 初始化MCP服务
mcp = FastMCP("enhanced-bio-invasion-server")

@mcp.tool()
async def enhanced_natural_language_query(
    query_text: str,
    db_name: Optional[str] = None,
    collection_name: Optional[str] = None,
    limit: int = 5,
    use_reranker: bool = True,
    enhance_output: bool = False
) -> Dict:
    """增强版自然语言查询MongoDB数据库
    
    完整流程：用户输入文本 → 文本转向量 → 向量查询 → 重排序 → 源文本查找 → 优化输出
    
    Args:
        query_text (str): 自然语言查询文本
        db_name (str, optional): 数据库名
        collection_name (str, optional): 集合名
        limit (int): 返回结果数量（默认5）
        use_reranker (bool): 是否使用重排序模型（默认True）
        enhance_output (bool): 是否使用AI优化输出（默认False）
    
    Returns:
        Dict: 包含完整查询结果的字典
    """
    try:
        # 参数校验
        if not query_text.strip():
            return {"status": "error", "msg": "必须提供query_text参数"}
        
        if mongo_col is None:
            return {"status": "error", "msg": "MongoDB连接未初始化"}
        
        # 确定使用的数据库和集合
        target_db = db_name or os.getenv('MONGO_DB_NAME', '中国生物入侵研究')
        target_collection = collection_name or os.getenv('MONGO_COLLECTION', '生物入侵研究')
        
        # 获取查询向量
        try:
            query_vector = text_to_vector(query_text)
            logger.info(f"✅ 文本转向量成功：{query_text[:50]}...")
        except Exception as e:
            return {"status": "error", "msg": f"文本转向量失败: {str(e)}"}
        
        # 切换到目标数据库和集合
        try:
            client = MongoClient(os.getenv('MONGO_URI', 'mongodb://localhost:27017/'))
            db = client[target_db]
            collection = db[target_collection]
            logger.info(f"✅ 切换到数据库：{target_db}.{target_collection}")
        except Exception as e:
            return {"status": "error", "msg": f"数据库切换失败: {str(e)}"}
        
        # 执行向量查询 - 直接查询MongoDB中的向量数据
        results = []
        for doc in collection.find({}):
            if 'data' in doc:
                similarity = cosine_similarity(np.array(doc['data']), query_vector)
                results.append((doc, similarity))
        
        # 初始排序
        results.sort(key=lambda x: x[1], reverse=True)
        
        # 重排序
        if use_reranker and reranker_tokenizer and reranker_model and query_text:
            results = rerank_results(query_text, results, reranker_tokenizer, reranker_model)
            logger.info("✅ 重排序完成")
        
        # 查找对应的原文档分段
        original_docs = []
        for doc, sim in results[:limit]:
            # 从向量文档的source字段提取编号
            vector_source = doc.get('source', '')
            if vector_source.startswith('ias_cn_') and vector_source.endswith('.npy'):
                # 提取编号并构建对应的原文档分段文件名
                chunk_number = vector_source.replace('ias_cn_', '').replace('.npy', '')
                try:
                    chunk_number = int(chunk_number)
                    # 查找对应的原文档分段
                    original_doc = collection.find_one({
                        'chunk_number': chunk_number,
                        'file_type': 'markdown_chunk'
                    })
                    original_docs.append((doc, sim, original_doc))
                except ValueError:
                    original_docs.append((doc, sim, None))
            else:
                original_docs.append((doc, sim, None))
        
        # 构建结果
        formatted_results = []
        for i, (doc, sim, original_doc) in enumerate(original_docs, 1):
            result_data = {
                "rank": i,
                "similarity_score": float(sim),
                "relevance_level": "高度相关" if sim > 0.8 else "中等相关" if sim > 0.5 else "低相关",
                "vector_metadata": doc['metadata'],
                "original_content_available": original_doc is not None
            }
            
            if original_doc:
                result_data.update({
                    "chunk_number": original_doc.get('chunk_number'),
                    "source_file": original_doc.get('source'),
                    "content_preview": original_doc['content'][:200] + "..." if len(original_doc['content']) > 200 else original_doc['content'],
                    "content_length": len(original_doc['content'])
                })
            
            formatted_results.append(result_data)
        
        # AI优化输出
        enhanced_report = None
        if enhance_output and query_text and len(results) > 0:
            enhanced_report = enhance_results_with_llm(query_text, original_docs[:limit])
        
        # 返回完整结果
        response_data = {
            "status": "success",
            "query_info": {
                "query_text": query_text,
                "database": target_db,
                "collection": target_collection,
                "total_results": len(results),
                "returned_results": len(formatted_results)
            },
            "results": formatted_results
        }
        
        if enhanced_report:
            response_data["enhanced_report"] = enhanced_report
        
        return response_data
        
    except Exception as e:
        logger.error(f"查询失败: {str(e)}")
        return {"status": "error", "msg": str(e)}
    finally:
        if 'client' in locals():
            client.close()

# -------------------------- 资源清理 --------------------------
async def cleanup_resources():
    """释放全局资源"""
    logger.info("🛑 开始释放资源")
    # 关闭MongoDB连接
    if mongo_client:
        mongo_client.close()
        logger.info("✅ MongoDB连接已关闭")

# -------------------------- 主函数 --------------------------
if __name__ == "__main__":
    try:
        # 启动MCP服务器
        logger.info("🚀 启动增强版生物入侵MCP服务器...")
        print("🚀 增强版生物入侵MCP服务器启动中...")
        mcp.run(transport="stdio")
    except KeyboardInterrupt:
        logger.info("\n⚠️  收到中断信号（Ctrl+C）")
        print("\n⚠️  收到中断信号（Ctrl+C）")
    except Exception as e:
        logger.error(f"服务器启动失败: {str(e)}")
        print(f"❌ 服务器启动失败: {str(e)}")
    finally:
        # 释放资源
        asyncio.run(cleanup_resources())
        logger.info("✅ 服务已完全停止")
        print("✅ 服务已完全停止")
