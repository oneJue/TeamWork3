# Accessing arXiv Data

arXiv 提供开放的预印本论文 API 接口，支持基于关键词、主题分类、作者等进行查询。

## 参考资料
- arXiv API 官方文档：https://info.arxiv.org/help/api/index.html
- 使用 arXiv API 的 Python 示例：https://github.com/lukasschwab/arxiv.py
- arXiv 主题分类：https://arxiv.org/category_taxonomy

## 应用说明
系统每日定时调用 arXiv API，获取新增论文，并对其元数据（title, abstract, categories, authors）进行解析与语义匹配。
