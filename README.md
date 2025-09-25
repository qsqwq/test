# 增强版生物入侵MCP服务器

这是一个专门为生物入侵研究设计的增强版MongoDB自然语言查询MCP服务器。基于FastMCP框架，提供完整的自然语言查询流程，包括文本转向量、向量查询、智能重排序、原文档查找和AI增强输出等功能。

## 功能特性

- ✅ **增强版自然语言查询**: 完整的自然语言查询流程，支持文本转向量、向量查询、重排序、原文档查找
- ✅ **智能重排序**: 集成Qwen3-Reranker-4B模型进行结果优化
- ✅ **原文档查找**: 自动查找对应的原文档分段内容
- ✅ **AI增强输出**: 支持DeepSeek API进行结果优化和报告生成
- ✅ **灵活配置**: 支持数据库和集合切换
- ✅ **向量索引支持**: 支持MongoDB向量索引查询，提高查询效率
- ✅ **备用嵌入模型**: 主模型失败时自动回退到备用模型
- ✅ **资源管理**: 自动管理数据库连接和HTTP会话

## 安装依赖

```bash
pip install -r requirements.txt
```

## 环境配置

创建 `.env` 文件并配置以下环境变量：

```env
# MongoDB配置
MONGO_URI=mongodb://localhost:27017/
MONGO_DB_NAME=中国生物入侵研究
MONGO_COLLECTION=生物入侵研究

# DeepSeek API配置（可选，用于结果优化）
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_API_URL=https://api.deepseek.com/v1/chat/completions

# 重排序模型配置（可选，默认使用Qwen3-Reranker-4B）
RERANKER_MODEL=Qwen3-Reranker-4B

# 嵌入模型配置（可选，默认使用qwen3-embedding:8b）
EMBEDDING_MODEL=qwen3-embedding:8b

# 服务器环境
MCP_ENV=development
```

## MCP配置

在MCP客户端配置文件中添加：

```json
{
  "mcpServers": {
    "enhanced-bio-invasion-server": {
      "command": "python",
      "args": [
        "server.py"
      ],
      "env": {
        "PYTHONPATH": "."
      }
    }
  }
}
```

## MCP功能

### 工具
- **`enhanced_natural_language_query`**: 增强版自然语言查询工具

### 资源
- **`config://app-version`**: 获取应用版本信息
- **`config://server-status`**: 获取服务器状态信息
- **`data://species/count`**: 获取数据库中物种总数
- **`data://species/{species_id}`**: 根据ID获取特定物种信息
- **`info://supported-queries`**: 获取支持的查询类型列表

### 提示模板
- **`enhanced_species_query`**: 增强版自然语言查询入侵物种详细信息
- **`enhanced_control_measures`**: 增强版查询入侵物种防治措施

## 工具使用

### 增强版自然语言查询工具

**工具名称**: `enhanced_natural_language_query`

**参数说明**:
- `query_text` (str, 必需): 自然语言查询文本
- `db_name` (str, 可选): 数据库名，默认为环境变量配置
- `collection_name` (str, 可选): 集合名，默认为环境变量配置
- `limit` (int): 返回结果数量，默认5
- `use_reranker` (bool): 是否使用重排序模型，默认True
- `enhance_output` (bool): 是否使用AI优化输出，默认False

**使用示例**:

```python
# 基本文本查询
result = await mcp.call_tool(
    "enhanced_natural_language_query",
    {
        "query_text": "红火蚁的入侵路径",
        "limit": 5,
        "use_reranker": True
    }
)

# 启用AI增强输出的查询
result = await mcp.call_tool(
    "enhanced_natural_language_query",
    {
        "query_text": "生物入侵防治措施",
        "limit": 5,
        "enhance_output": True
    }
)

# 切换数据库查询
result = await mcp.call_tool(
    "enhanced_natural_language_query",
    {
        "query_text": "外来物种风险评估",
        "db_name": "自定义数据库",
        "collection_name": "自定义集合",
        "use_reranker": True,
        "enhance_output": True
    }
)
```

## 返回结果格式

成功查询返回的数据结构：

```json
{
  "status": "success",
  "query_info": {
    "query_text": "查询文本",
    "database": "数据库名",
    "collection": "集合名",
    "total_results": 总结果数,
    "returned_results": 返回结果数
  },
  "results": [
    {
      "rank": 排名,
      "similarity_score": 相似度分数,
      "relevance_level": "相关程度",
      "vector_metadata": "向量文档元数据",
      "original_content_available": "是否找到原文档",
      "chunk_number": "分段编号",
      "source_file": "来源文件",
      "content_preview": "内容预览",
      "content_length": "内容长度"
    }
  ],
  "enhanced_report": "AI优化报告（如果启用）"
}
```

## 错误处理

工具会返回标准的错误格式：

```json
{
  "status": "error",
  "msg": "错误描述信息"
}
```

## 测试服务器

可以使用MCP检查器验证服务器功能：

```bash
# 安装MCP检查器
pip install mcp-inspector

# 运行检查器
mcp-inspector --command python --args server.py
```

## 启动服务器

直接运行服务器：

```bash
python server.py
```

## 技术架构

- **向量生成**: 使用Qwen3-Embedding-8B模型
- **重排序**: 使用Qwen3-Reranker-4B模型
- **AI增强**: 集成DeepSeek API
- **数据库**: MongoDB向量搜索
- **框架**: FastMCP v2

## 注意事项

1. 确保MongoDB服务正常运行
2. 向量模型需要本地安装或可访问
3. DeepSeek API为可选功能，不配置也可正常使用
4. 重排序模型需要GPU支持以获得最佳性能

## 许可证

本项目基于MIT许可证开源。
