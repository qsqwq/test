import os
import json
import requests
import torch
import asyncio
from typing import Dict, List, Optional
from pymongo import MongoClient
from dotenv import load_dotenv
from transformers import AutoModelForSequenceClassification, AutoTokenizer
# 1. 按MCP 1.13.1规范导入FastMCP
from mcp.server import FastMCP

# -------------------------- 环境配置与核心处理器 --------------------------
load_dotenv()

class MongoNLQueryHandler:
    """MongoDB自然语言查询核心处理器"""
    def __init__(self):
        # MongoDB配置
        self.mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
        self.db_name = os.getenv("MONGO_DB_NAME", "中国生物入侵研究")
        self.col_name = os.getenv("MONGO_COLLECTION", "生物入侵研究")
        self._init_mongo()

        # 模型配置
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "dengcao/Qwen3-Embedding-8B:Q5_K_M")
        self.reranker_model_name = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-large")
        self.reranker_model = None
        self.reranker_tokenizer = None
        self._init_reranker()

        # DeepSeek API配置
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        self.deepseek_api_url = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions")

    def _init_mongo(self):
        """初始化MongoDB连接"""
        try:
            self.client = MongoClient(self.mongo_uri)
            self.db = self.client[self.db_name]
            self.collection = self.db[self.col_name]
            self.client.admin.command("ping")
            
            print(f"✅ 连接MongoDB成功：{self.db_name} -> {self.col_name}")
        except Exception as e:
            raise RuntimeError(f"MongoDB连接失败：{str(e)}")

    def _init_reranker(self):
        """初始化重排序模型"""
        try:
            self.reranker_tokenizer = AutoTokenizer.from_pretrained(self.reranker_model_name)
            self.reranker_model = AutoModelForSequenceClassification.from_pretrained(self.reranker_model_name)
            self.reranker_model.eval()
            print(f"✅ 加载重排序模型成功：{self.reranker_model_name}")
        except Exception as e:
            print(f"⚠️  重排序模型加载失败：{str(e)}（将使用原生向量排序）")

    async def get_embedding(self, text: str) -> List[float]:
        """生成文本嵌入向量"""
        try:
            response = requests.post(
                "http://localhost:11434/api/embeddings",
                json={"model": self.embedding_model, "prompt": text},
                timeout=10
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except Exception as e:
            raise RuntimeError(f"向量生成失败：{str(e)}")

    async def query(self, query_text: str, limit: int = 5, use_reranker: bool = True) -> Dict:
        """自然语言查询MongoDB"""
        # 生成查询向量
        query_vec = await self.get_embedding(query_text)
        
        # 向量搜索
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "queryVector": query_vec,
                    "path": "embedding",
                    "limit": limit * 2
                }
            },
            {"$project": {"embedding": 0, "vector_score": {"$meta": "vectorSearchScore"}}}
        ]
        
        results = list(self.collection.aggregate(pipeline))
        
        # 重排序
        if use_reranker and self.reranker_model and len(results) > 1:
            doc_contents = [str(doc.get("content", "")) for doc in results]
            pairs = [[query_text, c] for c in doc_contents]
            
            with torch.no_grad():
                inputs = self.reranker_tokenizer(
                    pairs, padding=True, truncation=True, return_tensors="pt", max_length=512
                )
                scores = self.reranker_model(**inputs).logits.squeeze().tolist()
            
            for idx, score in enumerate(scores):
                results[idx]["rerank_score"] = float(score)
            results.sort(key=lambda x: x["rerank_score"], reverse=True)
            results = results[:limit]
        
        return {
            "status": "success",
            "query": query_text,
            "count": len(results),
            "results": results
        }

    async def enhance_with_deepseek(self, query: str, result: Dict) -> Dict:
        """用DeepSeek增强结果"""
        if not self.deepseek_api_key:
            result["enhance_msg"] = "未配置DeepSeek API Key，跳过增强"
            return result
        
        try:
            prompt = f"为查询「{query}」的结果添加1-2句解释（新增enhanced_info字段）：\n{json.dumps(result['results'])}"
            response = requests.post(
                self.deepseek_api_url,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.deepseek_api_key}"
                },
                json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3
                },
                timeout=15
            )
            response.raise_for_status()
            result["results"] = json.loads(response.json()["choices"][0]["message"]["content"])
            result["enhance_msg"] = "增强成功"
            return result
        except Exception as e:
            result["enhance_msg"] = f"增强失败：{str(e)}"
            return result

# -------------------------- 2. 实例化MCP服务器（按1.13.1规范） --------------------------
app = FastMCP("mongodb-bio-invasion-server")  # 使用FastMCP替代Server

# -------------------------- 3. 初始化处理器实例 --------------------------
mongo_handler = MongoNLQueryHandler()

# -------------------------- 4. 用@app.tool装饰器注册工具（核心修改） --------------------------
@app.tool(
    name="text_to_vector",
    description="将文本转换为768维嵌入向量（基于Qwen3-Embedding模型）",
    structured_output=True
)
async def text_to_vector(text: str) -> str:
    """
    将输入文本转换为向量表示
    :param text: 需转换的文本（建议≤512字符）
    :return: 包含向量数据的JSON字符串
    """
    try:
        vector = await mongo_handler.get_embedding(text)
        return json.dumps({
            "status": "success",
            "text": text,
            "vector": vector,
            "dimension": len(vector)
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            "status": "error",
            "error": str(e)
        }, ensure_ascii=False)

@app.tool(
    name="query_mongo_nl",
    description="用自然语言查询生物入侵研究数据库，支持向量搜索和结果重排序",
    structured_output=True
)
async def query_mongo_nl(
    query_text: str,
    limit: int = 5,
    use_reranker: bool = True,
    enhance: bool = False
) -> str:
    """
    自然语言查询MongoDB数据库
    :param query_text: 自然语言查询语句
    :param limit: 返回结果数量（1-20）
    :param use_reranker: 是否使用BGE模型重排序
    :param enhance: 是否用DeepSeek增强结果
    :return: 包含查询结果的JSON字符串
    """
    try:
        result = await mongo_handler.query(query_text, limit, use_reranker)
        if enhance:
            result = await mongo_handler.enhance_with_deepseek(query_text, result)
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            "status": "error",
            "query": query_text,
            "error": str(e)
        }, ensure_ascii=False)

@app.tool(
    name="list_tools",
    description="返回当前MCP服务器提供的所有工具列表及参数说明"
)
def list_tools() -> str:
    """返回可用工具列表"""
    tools = [
        {
            "name": "text_to_vector",
            "description": "将文本转换为768维嵌入向量",
            "parameters": [
                {"name": "text", "type": "string", "required": True, "desc": "输入文本"}
            ]
        },
        {
            "name": "query_mongo_nl",
            "description": "自然语言查询MongoDB数据库",
            "parameters": [
                {"name": "query_text", "type": "string", "required": True, "desc": "查询语句"},
                {"name": "limit", "type": "int", "default": 5, "desc": "结果数量"},
                {"name": "use_reranker", "type": "bool", "default": True, "desc": "是否重排序"},
                {"name": "enhance", "type": "bool", "default": False, "desc": "是否AI增强"}
            ]
        },
        {
            "name": "list_tools",
            "description": "查看所有可用工具",
            "parameters": []
        }
    ]
    return json.dumps(tools, ensure_ascii=False)

# -------------------------- 5. 按规范启动服务器 --------------------------
async def run_server():
    print("\n🚀 MCP服务器启动中...")
    print(f"服务名称: {app.name}")
    print("可用工具: text_to_vector, query_mongo_nl, list_tools")
    await app.run_stdio_async()  # 使用MCP 1.13.1的标准启动方式

if __name__ == "__main__":
    asyncio.run(run_server())  # 异步运行服务器
