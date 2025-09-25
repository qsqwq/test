import os
import json
import asyncio
import logging
import numpy as np
import requests
import torch
import aiohttp
import ollama
from typing import Dict, List, Optional, Tuple
from pymongo import MongoClient
from dotenv import load_dotenv
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 使用FastMCP服务器
from mcp.server.fastmcp import FastMCP

# -------------------------- 基础配置 --------------------------
# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)
logger = logging.getLogger("bio-invasion-mcp")

# 加载环境变量
load_dotenv()
ENV = os.getenv("MCP_ENV", "development")

# -------------------------- 全局资源初始化 --------------------------
# 1. 异步HTTP会话（懒加载）
aiohttp_session = None

# 2. MongoDB连接
def init_mongo():
    # 使用新的环境变量命名约定
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    db_name = os.getenv("MONGO_DB_NAME", "中国生物入侵研究")
    col_name = os.getenv("MONGO_COLLECTION", "生物入侵研究")
    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")  # 验证连接
        db = client[db_name]
        col = db[col_name]
        logger.info(f"✅ MongoDB连接成功：{db_name}.{col_name}")
        return client, col
    except Exception as e:
        logger.error(f"❌ MongoDB初始化失败：{str(e)}")
        # 不抛出异常，允许服务器在没有MongoDB的情况下启动（用于测试）
        return None, None

mongo_client, mongo_col = init_mongo()

# 3. 重排序模型（懒加载）
reranker_model = None
reranker_tokenizer = None  
def init_reranker():
    global reranker_model, reranker_tokenizer
    model_name = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-large")
    try:
        reranker_tokenizer = AutoTokenizer.from_pretrained(model_name) 
        reranker_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        reranker_model.eval()
        logger.info(f"✅ 重排序模型加载成功：{model_name}")
    except Exception as e:
        logger.warning(f"⚠️  重排序模型加载失败：{str(e)}（将用原生向量排序）")  

# 初始化重排序模型
init_reranker()

# -------------------------- 核心工具函数 --------------------------  
async def get_embedding(text: str) -> List[float]:
    """生成文本嵌入向量（基于本地Ollama模型）"""
    embedding_model = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:8b")
    
    try:
        # 直接使用本地Ollama模型
        response = ollama.embeddings(model=embedding_model, prompt=text)
        logger.info(f"✅ 使用本地嵌入模型成功: {embedding_model}")
        return response["embedding"]
    except Exception as e:
        logger.warning(f"⚠️  主嵌入模型 {embedding_model} 失败: {str(e)}，尝试备用模型")
        
        # 回退到nomic-embed-text模型
        try:
            response = ollama.embeddings(model="nomic-embed-text", prompt=text)
            logger.info(f"✅ 使用备用嵌入模型成功")
            return response["embedding"]
        except Exception as fallback_error:
            raise RuntimeError(f"所有嵌入模型都失败: 主模型错误 - {str(e)}, 备用模型错误 - {str(fallback_error)}")

async def enhance_with_deepseek(query: str, results: List[Dict]) -> List[Dict]:
    """用DeepSeek增强结果（添加专业解释）"""
    global aiohttp_session
    # 懒加载aiohttp会话
    if aiohttp_session is None:
        aiohttp_session = aiohttp.ClientSession()
    
    api_key = os.getenv("DEEPSEEK_API_KEY")
    api_url = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions")
    if not api_key:
        logger.warning("⚠️  未配置DeepSeek API Key，跳过增强")
        return results
    
    try:
        prompt = f"""为生物入侵查询「{query}」的结果，每个条目新增1句专业解释（key为"enhanced_info"），仅返回JSON：
{json.dumps(results, ensure_ascii=False)}"""
        async with aiohttp_session.post(
            api_url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            },
            json={
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "max_tokens": 2048
            },
            timeout=15
        ) as resp:
            if resp.status != 200:
                raise RuntimeError(f"DeepSeek API status: {resp.status}")
            data = await resp.json()
            return json.loads(data["choices"][0]["message"]["content"])
    except Exception as e:
        logger.error(f"❌ DeepSeek增强失败：{str(e)}")
        return results

# -------------------------- MCP服务器初始化 --------------------------
# 初始化MCP服务
mcp = FastMCP("bio-invasion-mongo-server")

# -------------------------- 核心工具函数（从enhanced_query_server.py导入） --------------------------
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
    client = None
    try:
        # 参数校验
        if not query_text.strip():
            return {"status": "error", "msg": "必须提供query_text参数"}
        
        # 确定使用的数据库和集合
        target_db = db_name or os.getenv('MONGO_DB_NAME', '中国生物入侵研究')
        target_collection = collection_name or os.getenv('MONGO_COLLECTION', '生物入侵研究')
        
        # 获取查询向量
        try:
            query_vector = await get_embedding(query_text)
            logger.info(f"✅ 文本转向量成功：{query_text[:50]}...")
        except Exception as e:
            logger.error(f"文本转向量失败: {str(e)}")
            return {"status": "error", "msg": f"文本转向量失败: {str(e)}"}
        
        # 创建新的MongoDB连接
        try:
            client = MongoClient(os.getenv('MONGO_URI', 'mongodb://localhost:27017/'))
            db = client[target_db]
            collection = db[target_collection]
            logger.info(f"✅ 连接到数据库：{target_db}.{target_collection}")
        except Exception as e:
            logger.error(f"数据库连接失败: {str(e)}")
            return {"status": "error", "msg": f"数据库连接失败: {str(e)}"}
        
        # 执行向量查询 - 使用MongoDB向量索引（如果可用）
        results = []
        try:
            # 首先检查数据库中有多少文档
            total_docs = collection.count_documents({})
            logger.info(f"📊 数据库中总文档数: {total_docs}")
            
            # 检查前几个文档的结构
            sample_docs = list(collection.find().limit(3))
            logger.info(f"📋 样本文档结构: {[list(doc.keys()) for doc in sample_docs]}")
            
            if "vector_index" in collection.index_information():
                # 使用向量索引进行高效查询
                pipeline = [
                    {
                        "$vectorSearch": {
                            "index": "vector_index",
                            "queryVector": query_vector,
                            "path": "embedding",
                            "limit": limit * 2  # 取2倍用于重排序
                        }
                    },
                    {"$project": {"embedding": 0, "vector_score": {"$meta": "vectorSearchScore"}}}
                ]
                vector_results = list(collection.aggregate(pipeline))
                
                # 转换为与enhanced_query_server.py兼容的格式
                for doc in vector_results:
                    similarity = doc.get("vector_score", 0.0)
                    # 确保文档有metadata字段，如果没有则创建空字典
                    if 'metadata' not in doc:
                        doc['metadata'] = {}
                    results.append((doc, similarity))
                logger.info(f"✅ 使用向量索引查询成功，找到 {len(results)} 个结果")
            else:
                # 回退到线性扫描（兼容enhanced_query_server.py的逻辑）
                logger.warning("⚠️  无vector_index，使用基础查询")
                docs_with_vector = 0
                for doc in collection.find({}):
                    # 检查文档是否有data字段（向量数据）
                    if 'data' in doc:
                        try:
                            docs_with_vector += 1
                            # 确保query_vector是numpy数组
                            query_vec = np.array(query_vector)
                            doc_vec = np.array(doc['data'])
                            similarity = cosine_similarity(query_vec, doc_vec)
                            results.append((doc, similarity))
                        except Exception as e:
                            logger.warning(f"文档 {doc.get('source', 'unknown')} 向量计算失败: {str(e)}")
                            continue  # 跳过这个文档，继续处理下一个
                
                logger.info(f"📊 有data字段（向量）的文档数: {docs_with_vector}")
                logger.info(f"✅ 基础查询完成，找到 {len(results)} 个结果")
        except Exception as e:
            logger.error(f"MongoDB查询失败: {str(e)}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            return {"status": "error", "msg": f"MongoDB查询失败: {str(e)}"}
        
        # 初始排序
        results.sort(key=lambda x: x[1], reverse=True)
        
        # 重排序
        if use_reranker and reranker_tokenizer and reranker_model and query_text:
            try:
                results = rerank_results(query_text, results, reranker_tokenizer, reranker_model)
                logger.info("✅ 重排序完成")
            except Exception as e:
                logger.warning(f"重排序失败，使用原始排序: {str(e)}")
        
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
                "vector_metadata": doc.get('metadata', {}),
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
            try:
                enhanced_report = enhance_results_with_llm(query_text, original_docs[:limit])
            except Exception as e:
                logger.warning(f"AI优化输出失败: {str(e)}")
        
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
        import traceback
        logger.error(f"详细错误信息: {traceback.format_exc()}")
        return {"status": "error", "msg": str(e)}
    finally:
        # 确保数据库连接被正确关闭
        if client:
            try:
                client.close()
                logger.info("✅ 数据库连接已关闭")
            except Exception as e:
                logger.warning(f"关闭数据库连接时出错: {str(e)}")

# -------------------------- MCP资源定义 --------------------------
@mcp.resource("config://app-version")
def get_app_version() -> str:
    """返回应用版本信息"""
    return "生物入侵MongoDB服务器 v1.0.0"

@mcp.resource("config://server-status")
def get_server_status() -> Dict:
    """返回服务器状态信息"""
    return {
        "status": "running",
        "environment": ENV,
        "mongo_connected": mongo_client is not None,
        "reranker_loaded": reranker_model is not None,
        "timestamp": asyncio.get_event_loop().time() if asyncio.get_event_loop().is_running() else 0
    }

@mcp.resource("data://species/count")
async def get_species_count() -> Dict:
    """返回数据库中物种总数"""
    try:
        if not mongo_col:
            return {"error": "MongoDB连接未初始化"}
        
        count = await asyncio.get_event_loop().run_in_executor(
            None, lambda: mongo_col.count_documents({})
        )
        return {"total_species": count}
    except Exception as e:
        return {"error": str(e)}

@mcp.resource("data://species/{species_id}")
async def get_species_by_id(species_id: str) -> Dict:
    """根据ID获取特定物种信息"""
    try:
        if not mongo_col:
            return {"error": "MongoDB连接未初始化"}
        
        species = await asyncio.get_event_loop().run_in_executor(
            None, lambda: mongo_col.find_one({"_id": species_id}, {"embedding": 0})
        )
        if species:
            return {"status": "found", "data": species}
        else:
            return {"status": "not_found", "species_id": species_id}
    except Exception as e:
        return {"error": str(e)}

@mcp.resource("info://supported-queries")
def get_supported_queries() -> List[str]:
    """返回支持的查询类型列表"""
    return [
        "物种基本信息查询",
        "入侵路径分析", 
        "分布范围查询",
        "防治措施查询",
        "风险评估查询"
    ]

# -------------------------- MCP提示模板 --------------------------
@mcp.prompt(name="enhanced_species_query", description="增强版自然语言查询入侵物种详细信息")
def enhanced_species_query_prompt(species_name: str) -> Dict:
    """生成增强版物种查询提示
    
    Args:
        species_name (str): 入侵物种名称（如"红火蚁"）
    
    Returns:
        Dict: MCP标准提示格式
    """
    return {
        "description": f"使用增强版自然语言查询{species_name}的生物入侵详细信息",
        "messages": [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"请使用增强版自然语言查询功能查询{species_name}的详细信息，包括：1.生物学特性；2.入侵历史；3.分布范围；4.识别特征；5.防治措施"
                }
            }
        ],
        "arguments": {
            "query_text": f"查询{species_name}的详细信息",
            "limit": 5,
            "use_reranker": True,
            "enhance_output": True
        }
    }

@mcp.prompt(name="enhanced_control_measures", description="增强版查询入侵物种防治措施")
def enhanced_control_measures_prompt(species_name: str) -> Dict:
    """生成增强版防治措施查询提示"""
    return {
        "description": f"使用增强版自然语言查询{species_name}的防治措施",
        "messages": [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"请使用增强版自然语言查询功能查询{species_name}的防治措施，包括：1.物理防治；2.化学防治；3.生物防治；4.综合策略"
                }
            }
        ],
        "arguments": {
            "query_text": f"查询{species_name}的防治措施",
            "limit": 5,
            "use_reranker": True,
            "enhance_output": True
        }
    }

# -------------------------- 资源清理 --------------------------
async def cleanup_resources():
    """释放全局资源"""
    logger.info("🛑 开始释放资源")
    # 关闭MongoDB连接
    if mongo_client:
        mongo_client.close()
        logger.info("✅ MongoDB连接已关闭")
    # 关闭aiohttp会话
    if aiohttp_session:
        await aiohttp_session.close()
        logger.info("✅ aiohttp会话已关闭")

# -------------------------- 主函数 --------------------------
if __name__ == "__main__":
    try:
        # 使用stdio传输方式启动MCP服务器（与客户端兼容）
        logger.info("🚀 启动MCP服务器...")
        mcp.run(transport="stdio")
    except KeyboardInterrupt:
        logger.info("\n⚠️  收到中断信号（Ctrl+C）")
    finally:
        # 释放资源
        asyncio.run(cleanup_resources())
        logger.info("✅ 服务已完全停止")
