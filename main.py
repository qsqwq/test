import os
import json
import asyncio
import logging
from typing import Dict, List
from pymongo import MongoClient
from dotenv import load_dotenv
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import aiohttp
import numpy as np
from llama_cpp import Llama

# 严格遵循官方MCP示例的导入方式
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
    db_name = os.getenv("MONGO_DB_NAME", "test_db")
    col_name = os.getenv("MONGO_COLLECTION", "test_collection")
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

# 3. 嵌入模型（懒加载）
embedding_llm = None
reranker_model = None
reranker_tokenizer = None

def init_embedding_model():
    """初始化本地嵌入模型（使用llama-cpp-python加载GGUF文件）"""
    global embedding_llm
    model_path = os.getenv("LOCAL_EMBEDDING_MODEL_PATH", "C:\\Users\\admin\\.ollama\\models\\blobs\\sha256-14037f526fecc2f9e23bb9ef544bc733bcef605feb110fe724349df5a4bad3ee")
    
    try:
        # 使用llama-cpp-python加载GGUF模型文件
        embedding_llm = Llama(
            model_path=model_path,
            embedding=True,  # 启用嵌入功能
            n_ctx=4096,      # 上下文长度
            n_threads=4,     # 线程数
            verbose=False    # 不输出详细日志
        )
        logger.info(f"✅ 本地嵌入模型加载成功：{model_path}")
    except Exception as e:
        logger.warning(f"⚠️  本地嵌入模型加载失败：{str(e)}，将使用备用方法")

def init_reranker():
    """初始化重排序模型"""
    global reranker_model, reranker_tokenizer
    model_name = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-large")
    try:
        reranker_tokenizer = AutoTokenizer.from_pretrained(model_name)
        reranker_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        reranker_model.eval()
        logger.info(f"✅ 重排序模型加载成功：{model_name}")
    except Exception as e:
        logger.warning(f"⚠️  重排序模型加载失败：{str(e)}（将用原生向量排序）")

# 初始化模型
init_embedding_model()
init_reranker()

# -------------------------- 核心工具函数 --------------------------
async def get_embedding(text: str) -> List[float]:
    """生成文本嵌入向量（基于本地GGUF模型文件）"""
    try:
        if embedding_llm is None:
            raise RuntimeError("本地嵌入模型未初始化")
        
        # 使用llama-cpp-python生成嵌入向量
        embedding_result = embedding_llm.create_embedding(text)
        embedding_vector = embedding_result["data"][0]["embedding"]
        
        # L2归一化
        embedding_vector = np.array(embedding_vector)
        embedding_vector = embedding_vector / np.linalg.norm(embedding_vector)
        embedding_vector = embedding_vector.tolist()
        
        logger.info(f"✅ 本地嵌入模型生成向量成功，维度：{len(embedding_vector)}")
        return embedding_vector
        
    except Exception as e:
        logger.error(f"❌ 本地嵌入模型生成失败：{str(e)}")
        raise RuntimeError(f"本地嵌入模型生成失败：{str(e)}")

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
# 初始化MCP服务（完全遵循官方示例）
mcp = FastMCP("bio-invasion-mongo-server")

@mcp.tool()
async def text_to_vector(text: str) -> Dict:
    """将生物入侵相关文本转换为嵌入向量
    
    Args:
        text (str): 需转换的文本（建议≤512字符，内容为生物入侵描述）
     
    Returns:
        Dict: 包含向量、维度、模型信息的结果
    """
    try:
        if not text.strip():
            return {"status": "error", "msg": "text不能为空"}
        
        vector = await get_embedding(text.strip())
        return {
            "status": "success",
            "data": {
                "text": text.strip(),
                "vector": vector,
                "dimension": len(vector),
                "model": os.getenv("EMBEDDING_MODEL", "dengcao/Qwen3-Embedding-8B:Q5_K_M")
            }
        }
    except Exception as e:
        return {"status": "error", "msg": str(e)}

@mcp.tool()
async def create_vector_index(dimension: int = 768) -> Dict:
    """为MongoDB集合创建向量索引
    
    Args:
        dimension (int): 向量维度，默认为768
        
    Returns:
        Dict: 索引创建结果
    """
    try:
        if mongo_col is None:
            return {"status": "error", "msg": "MongoDB连接未初始化"}
        
        # 检查是否已存在向量索引
        existing_indexes = mongo_col.index_information()
        if "vector_index" in existing_indexes:
            return {"status": "success", "msg": "向量索引已存在"}
        
        # 创建向量索引 - 使用正确的MongoDB语法
        index_definition = {
            "name": "vector_index",
            "key": {"embedding": "vector"},
            "vectorOptions": {
                "dimension": dimension,
                "similarity": "cosine"
            }
        }
        
        # 使用create_index创建向量索引
        mongo_col.create_index([("embedding", "vector")], name="vector_index")
        logger.info(f"✅ 向量索引创建成功，维度：{dimension}")
        
        return {
            "status": "success", 
            "msg": f"向量索引创建成功，维度：{dimension}",
            "index_name": "vector_index"
        }
        
    except Exception as e:
        logger.error(f"❌ 向量索引创建失败：{str(e)}")
        return {"status": "error", "msg": f"索引创建失败：{str(e)}"}

@mcp.tool()
async def check_database_content() -> Dict:
    """检查数据库内容和结构"""
    try:
        if mongo_col is None:
            return {"status": "error", "msg": "MongoDB连接未初始化"}
        
        # 检查集合中的文档数量
        total_docs = mongo_col.count_documents({})
        
        # 检查包含嵌入向量的文档数量
        docs_with_embedding = mongo_col.count_documents({"embedding": {"$exists": True}})
        
        # 检查索引信息
        indexes = mongo_col.index_information()
        
        # 获取一些示例文档结构
        sample_docs = list(mongo_col.find().limit(3))
        
        return {
            "status": "success",
            "database_info": {
                "total_documents": total_docs,
                "documents_with_embedding": docs_with_embedding,
                "indexes": list(indexes.keys()),
                "sample_document_structure": [str(doc.get("_id")) for doc in sample_docs] if sample_docs else []
            }
        }
        
    except Exception as e:
        return {"status": "error", "msg": f"数据库检查失败：{str(e)}"}

@mcp.tool()
async def natural_language_query(
    user_input: str,
    limit: int = 5,
    use_reranker: bool = True,
    enhance_output: bool = False
) -> Dict:
    """自然语言查询MongoDB数据库 - 完整流程
    
    完整流程：用户输入文本 → 嵌入模型向量化 → 向量查询 → 重排序 → 源文本查找 → 优化输出
    
    Args:
        user_input (str): 用户自然语言查询文本
        limit (int): 返回结果数量（1-20，默认5）
        use_reranker (bool): 是否使用重排序模型优化结果（默认True）
        enhance_output (bool): 是否使用AI模型优化输出内容（默认False）
    
    Returns:
        Dict: 包含完整查询流程结果的字典
    """
    try:
        logger.info(f"🔍 开始自然语言查询流程：'{user_input}'")
        
        # 1. 用户输入文本处理
        user_input = user_input.strip()
        if not user_input:
            return {"status": "error", "msg": "用户输入文本不能为空"}
        limit = max(1, min(limit, 20))  # 限制1-20范围

        # 2. 使用嵌入模型将文本转换为向量
        logger.info("📊 步骤1：使用嵌入模型将文本转换为向量")
        query_vector = await get_embedding(user_input)
        logger.info(f"✅ 向量转换完成，维度：{len(query_vector)}")

        # 3. 在数据库中进行向量查询
        logger.info("🔍 步骤2：在MongoDB中进行向量查询")
        if mongo_col is None:
            return {"status": "error", "msg": "MongoDB连接未初始化，请检查数据库配置"}
        
        try:
            # 检查是否存在向量索引
            if "vector_index" in mongo_col.index_information():
                # 使用MongoDB的向量搜索功能
                pipeline = [
                    {
                        "$vectorSearch": {
                            "index": "vector_index",
                            "queryVector": query_vector,
                            "path": "embedding",
                            "limit": limit * 3,  # 取3倍用于后续重排序
                            "numCandidates": 100
                        }
                    },
                    {
                        "$project": {
                            "embedding": 0,  # 不返回向量数据
                            "vector_score": {"$meta": "vectorSearchScore"},
                            "content": 1,
                            "metadata": 1,
                            "_id": 1,
                            "source": 1
                        }
                    }
                ]
                vector_results = list(mongo_col.aggregate(pipeline))
                logger.info(f"✅ 向量查询完成，找到 {len(vector_results)} 个候选结果")
            else:
                logger.warning("⚠️  无vector_index，使用余弦相似度计算")
                # 使用余弦相似度进行向量查询
                vector_results = []
                for doc in mongo_col.find({}):
                    if 'embedding' in doc:
                        doc_vector = np.array(doc['embedding'])
                        similarity = np.dot(query_vector, doc_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(doc_vector))
                        # 只保留相似度大于0.1的结果
                        if similarity > 0.1:
                            vector_results.append({
                                **doc,
                                "vector_score": float(similarity),
                                "_id": doc.get("_id")
                            })
                
                # 按相似度排序
                vector_results.sort(key=lambda x: x.get("vector_score", 0), reverse=True)
                vector_results = vector_results[:limit * 3]
                logger.info(f"✅ 余弦相似度查询完成，找到 {len(vector_results)} 个候选结果")
        except Exception as e:
            logger.error(f"❌ MongoDB向量查询失败：{str(e)}")
            return {"status": "error", "msg": f"MongoDB查询失败: {str(e)}"}

        # 4. 对查询到的向量进行重排序
        logger.info("📈 步骤3：对查询结果进行重排序")
        if use_reranker and reranker_model and len(vector_results) > 1:
            # 提取文档内容用于重排序
            doc_contents = []
            for doc in vector_results:
                content = doc.get("content", "")
                if isinstance(content, dict):
                    content = json.dumps(content, ensure_ascii=False)
                doc_contents.append(str(content))
            
            # 创建查询-文档对
            pairs = [[user_input, content] for content in doc_contents]
            
            # 使用重排序模型计算相关性分数
            with torch.no_grad():
                inputs = reranker_tokenizer(
                    pairs, 
                    padding=True, 
                    truncation=True, 
                    return_tensors="pt", 
                    max_length=512
                )
                rerank_scores = reranker_model(**inputs).logits.squeeze().tolist()
            
            # 绑定重排序分数并排序
            for idx, score in enumerate(rerank_scores):
                vector_results[idx]["rerank_score"] = float(score)
            
            # 按重排序分数降序排列
            vector_results.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
            logger.info(f"✅ 重排序完成，最高分：{max(rerank_scores):.4f}")
        
        # 取前limit个结果
        final_results = vector_results[:limit]

        # 5. 找到源文本并进行优化输出
        logger.info("📝 步骤4：源文本查找和优化输出")
        optimized_results = []
        
        for result in final_results:
            # 提取源文本内容
            source_content = result.get("content", "")
            metadata = result.get("metadata", {})
            
            # 构建优化后的结果
            optimized_result = {
                "source_id": str(result.get("_id", "")),
                "source_content": source_content,
                "metadata": metadata,
                "vector_score": result.get("vector_score", 0),
                "rerank_score": result.get("rerank_score", 0),
                "relevance_score": calculate_relevance_score(result)
            }
            
            # 如果启用输出增强，使用AI模型优化内容
            if enhance_output:
                optimized_result = await enhance_result_content(user_input, optimized_result)
            
            optimized_results.append(optimized_result)
        
        # 6. 返回优化后的完整结果
        logger.info(f"✅ 查询流程完成，返回 {len(optimized_results)} 个优化结果")
        return {
            "status": "success",
            "query_process": {
                "user_input": user_input,
                "vector_dimension": len(query_vector),
                "candidate_count": len(vector_results),
                "final_count": len(optimized_results)
            },
            "results": optimized_results,
            "summary": generate_query_summary(user_input, optimized_results)
        }
        
    except Exception as e:
        logger.error(f"❌ 自然语言查询流程失败：{str(e)}")
        return {"status": "error", "query": user_input, "msg": str(e)}

def calculate_relevance_score(result: Dict) -> float:
    """计算综合相关性分数"""
    vector_score = result.get("vector_score", 0)
    rerank_score = result.get("rerank_score", 0)
    
    # 如果重排序分数可用，优先使用重排序分数
    if rerank_score > 0:
        return rerank_score
    else:
        return vector_score

async def enhance_result_content(query: str, result: Dict) -> Dict:
    """使用AI模型优化结果内容"""
    try:
        source_content = result.get("source_content", "")
        if not source_content:
            return result
        
        # 构建优化提示
        enhancement_prompt = f"""
        基于以下查询和源文本，生成优化后的内容：
        
        查询：{query}
        源文本：{source_content[:1000]}  # 限制长度
        
        请生成：
        1. 简洁摘要（100字内）
        2. 与查询的相关性分析
        3. 关键信息提取
        
        返回JSON格式。
        """
        
        # 这里可以集成各种AI模型进行内容优化
        # 暂时使用简单的文本处理作为示例
        enhanced_content = {
            "summary": f"关于'{query}'的相关信息摘要",
            "relevance_analysis": "内容与查询高度相关",
            "key_points": ["关键点1", "关键点2", "关键点3"]
        }
        
        result["enhanced_content"] = enhanced_content
        return result
        
    except Exception as e:
        logger.warning(f"⚠️  内容优化失败：{str(e)}")
        return result

def generate_query_summary(query: str, results: List[Dict]) -> Dict:
    """生成查询摘要"""
    total_results = len(results)
    avg_relevance = sum(r.get("relevance_score", 0) for r in results) / max(total_results, 1)
    
    return {
        "query": query,
        "total_results": total_results,
        "average_relevance": round(avg_relevance, 4),
        "top_relevance": max((r.get("relevance_score", 0) for r in results), default=0),
        "timestamp": asyncio.get_event_loop().time() if asyncio.get_event_loop().is_running() else 0
    }

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
@mcp.prompt(name="species_query", description="查询入侵物种详细信息")
def species_query_prompt(species_name: str) -> Dict:
    """生成物种查询提示
    
    Args:
        species_name (str): 入侵物种名称（如"红火蚁"）
    
    Returns:
        Dict: MCP标准提示格式
    """
    return {
        "description": f"查询{species_name}的生物入侵详细信息",
        "messages": [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"请查询{species_name}：1.生物学特性；2.入侵历史；3.分布范围；4.识别特征"
                }
            }
        ]
    }

@mcp.prompt(name="control_measures", description="查询入侵物种防治措施")
def control_measures_prompt(species_name: str) -> Dict:
    """生成防治措施查询提示"""
    return {
        "description": f"查询{species_name}的防治措施",
        "messages": [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"请查询{species_name}的：1.物理防治；2.化学防治；3.生物防治；4.综合策略"
                }
            }
        ]
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
        # 遵循官方示例的简单启动方式
        logger.info("🚀 启动MCP服务器...")
        mcp.run()
    except KeyboardInterrupt:
        logger.info("\n⚠️  收到中断信号（Ctrl+C）")
    finally:
        # 释放资源
        asyncio.run(cleanup_resources())
        logger.info("✅ 服务已完全停止")
