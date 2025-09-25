# MCP Inspector 手动传递环境变量完整命令行

## 基于您的项目配置的完整命令行

### 方案1：传递所有关键环境变量（推荐）
```bash
npx @modelcontextprotocol/inspector -e MONGO_URI=mongodb://localhost:27017/ -e MONGO_DB_NAME=中国生物入侵研究 -e MONGO_COLLECTION=生物入侵研究 -e EMBEDDING_MODEL=qwen3-embedding:8b -e RERANKER_MODEL=Qwen3-Reranker-4B -e MCP_ENV=development -e LOG_LEVEL=INFO python server.py
```

### 方案2：仅传递必需的环境变量（简化版）
```bash
npx @modelcontextprotocol/inspector -e MONGO_URI=mongodb://localhost:27017/ -e EMBEDDING_MODEL=qwen3-embedding:8b python server.py
```

### 方案3：使用enhanced_query_server.py（根据mcp.json配置）
```bash
npx @modelcontextprotocol/inspector -e MONGO_URI=mongodb://localhost:27017/ -e EMBEDDING_MODEL=qwen3-embedding:8b python enhanced_query_server.py
```

## 分步执行指南

### 步骤1：进入项目目录
```bash
cd d:\mcp_mongodb_bio_invasion
```

### 步骤2：选择适合的命令行执行

**如果您想使用server.py：**
```bash
npx @modelcontextprotocol/inspector -e MONGO_URI=mongodb://localhost:27017/ -e MONGO_DB_NAME=中国生物入侵研究 -e MONGO_COLLECTION=生物入侵研究 -e EMBEDDING_MODEL=qwen3-embedding:8b -e RERANKER_MODEL=Qwen3-Reranker-4B -e MCP_ENV=development -e LOG_LEVEL=INFO python server.py
```

**如果您想使用enhanced_query_server.py：**
```bash
npx @modelcontextprotocol/inspector -e MONGO_URI=mongodb://localhost:27017/ -e MONGO_DB_NAME=中国生物入侵研究 -e MONGO_COLLECTION=生物入侵研究 -e EMBEDDING_MODEL=qwen3-embedding:8b -e RERANKER_MODEL=Qwen3-Reranker-4B -e MCP_ENV=development -e LOG_LEVEL=INFO python enhanced_query_server.py
```

### 步骤3：在浏览器中访问MCP Inspector
打开浏览器访问：`http://localhost:5173`

## 环境变量说明

| 环境变量 | 值 | 作用 |
|---------|----|------|
| MONGO_URI | mongodb://localhost:27017/ | MongoDB连接地址 |
| MONGO_DB_NAME | 中国生物入侵研究 | 数据库名称 |
| MONGO_COLLECTION | 生物入侵研究 | 集合名称 |
| EMBEDDING_MODEL | qwen3-embedding:8b | 嵌入模型名称 |
| RERANKER_MODEL | Qwen3-Reranker-4B | 重排序模型名称 |
| MCP_ENV | development | 运行环境 |
| LOG_LEVEL | INFO | 日志级别 |

## 验证命令是否生效

### 在服务器代码中添加验证
在您的`server.py`或`enhanced_query_server.py`中添加以下代码来验证环境变量：

```python
import os

# 在服务器启动时检查环境变量
print("=== 环境变量验证 ===")
print(f"MONGO_URI: {os.getenv('MONGO_URI')}")
print(f"MONGO_DB_NAME: {os.getenv('MONGO_DB_NAME')}")
print(f"EMBEDDING_MODEL: {os.getenv('EMBEDDING_MODEL')}")
print("===================")
```

## 常见问题解决

### 问题1：路径包含中文或特殊字符
如果路径包含中文，确保使用正确的编码：

```bash
# 在Windows PowerShell中可能需要使用chcp命令设置编码
chcp 65001
npx @modelcontextprotocol/inspector -e MONGO_URI=mongodb://localhost:27017/ -e EMBEDDING_MODEL=qwen3-embedding:8b python server.py
```

### 问题2：权限问题
如果遇到权限问题，尝试以管理员身份运行命令行。

### 问题3：端口占用
如果5173端口被占用，MCP Inspector会自动选择其他端口，查看命令行输出获取实际地址。

## 一键执行脚本（可选）

创建一个批处理文件`start_mcp.bat`：
```batch
@echo off
cd /d d:\mcp_mongodb_bio_invasion
npx @modelcontextprotocol/inspector -e MONGO_URI=mongodb://localhost:27017/ -e EMBEDDING_MODEL=qwen3-embedding:8b python server.py
pause
```

通过以上命令行，您可以手动传递所有必要的环境变量来启动MCP Inspector调试您的服务器。
