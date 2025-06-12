# Recommendation Algorithms

本文系统使用语义匹配算法推荐与用户兴趣相符的论文。

## 可选算法
- 内容推荐：文献嵌入相似度（如 SPECTER + FAISS）
- 协同过滤（若有多人数据时）
- 混合推荐（内容 + 用户行为）
- Top-k 最近邻检索（ANN 搜索）

## 工具
- FAISS（Facebook AI 相似性搜索）: https://github.com/facebookresearch/faiss
- SentenceTransformers：https://www.sbert.net/
