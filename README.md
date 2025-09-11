# MongoDB 生物入侵研究 MCP 服务器

一个基于 Model Context Protocol (MCP) 的 MongoDB 自然语言查询服务器，专门用于生物入侵研究数据的智能检索和分析。

## 🚀 功能特性

- **自然语言查询**: 使用向量搜索技术实现自然语言到 MongoDB 查询的转换
- **智能重排序**: 集成 BGE 重排序模型提升搜索结果相关性
- **AI 增强**: 可选 DeepSeek API 对查询结果进行智能解释和增强
- **MCP 标准兼容**: 遵循 MCP 1.13.1 规范，可与各种 MCP 客户端集成

## 📦 安装依赖

```bash
# 使用 uv 包管理器安装
uv sync

# 或者使用 pip
pip install -e .
```

## ⚙️ 环境配置

创建 `.env` 文件并配置以下环境变量：

```env
# MongoDB 配置
MONGO_URI=mongodb://localhost:27017/
MONGO_DB_NAME=中国生物入侵研究
MONGO_COLLECTION=生物入侵研究

# 模型配置
EMBEDDING_MODEL=dengcao/Qwen3-Embedding-8B:Q5_K_M
RERANKER_MODEL=BAAI/bge-reranker-large

# DeepSeek API 配置（可选）
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_API_URL=https://api.deepseek.com/v1/chat/completions
```

## 🛠️ 可用工具

### 1. text_to_vector
将文本转换为 768 维嵌入向量

**参数:**
- `text` (string, 必需): 需要转换的文本（建议 ≤512 字符）

**示例:**
```json
{
  "text": "生物入侵对生态系统的影响"
}
```

### 2. query_mongo_nl
自然语言查询 MongoDB 数据库

**参数:**
- `query_text` (string, 必需): 自然语言查询语句
- `limit` (int, 可选, 默认=5): 返回结果数量（1-20）
- `use_reranker` (bool, 可选, 默认=true): 是否使用 BGE 模型重排序
- `enhance` (bool, 可选, 默认=false): 是否用 DeepSeek 增强结果

**示例:**
```json
{
  "query_text": "查找关于外来物种入侵的研究",
  "limit": 10,
  "use_reranker": true,
  "enhance": true
}
```

### 3. list_tools
查看所有可用工具列表及参数说明

## 🚀 启动服务器

```bash
# 使用 uv 运行
uv run python mcp_mongodb_server.py

# 或者直接运行
python mcp_mongodb_server.py
```

服务器启动后将通过 stdio 与 MCP 客户端通信。

## 📊 数据结构要求

MongoDB 集合需要包含以下字段以支持向量搜索：

```json
{
  "_id": ObjectId,
  "content": "文档内容文本",
  "embedding": [0.1, 0.2, 0.3, ...], // 768 维向量
  // 其他相关字段...
}
```

需要创建向量索引：
```javascript
db.collection.createIndex({
  "embedding": "vector"
}, {
  "name": "vector_index",
  "type": "vector",
  "dimension": 768,
  "similarity": "cosine"
})
```

## 🔧 技术栈

- **MCP Server**: `mcp[cli]>=1.13.1`
- **数据库**: `pymongo>=4.6.0`
- **机器学习**: `transformers>=4.37.0`, `torch>=2.1.0`
- **HTTP 请求**: `requests>=2.31.0`
- **环境管理**: `python-dotenv>=1.0.0`

## 🎯 使用场景

1. **科研人员**: 快速检索生物入侵相关研究文献
2. **政策制定者**: 分析入侵物种的影响和应对策略
3. **教育工作者**: 获取教学案例和研究资料
4. **环保组织**: 监测和评估生物入侵状况

## 🤝 贡献指南

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 🙏 致谢

- [Model Context Protocol](https://modelcontextprotocol.io) - 提供标准的 AI 工具协议
- [Hugging Face](https://huggingface.co) - 提供预训练模型
- [MongoDB](https://www.mongodb.com) - 提供向量搜索功能

## 📞 支持

如有问题或建议，请提交 [Issue](https://github.com/your-repo/issues) 或联系开发团队。

---

**注意**: 使用前请确保 MongoDB 服务器正常运行，并已配置正确的向量索引。
