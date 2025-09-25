"""
增强版生物入侵MCP服务器包
基于FastMCP框架，提供完整的自然语言查询MongoDB数据库功能
包括文本转向量、向量查询、智能重排序、原文档查找和AI增强输出
"""

from .server import mcp, enhanced_natural_language_query

__version__ = "1.0.0"
__all__ = ["mcp", "enhanced_natural_language_query"]
