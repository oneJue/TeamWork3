# User Interest Modeling

用户兴趣建模是精准推荐的基础，可以从显性（标签、关键词）或隐性（阅读记录、内容向量）进行建模。

## 参考方法
- TF-IDF + 余弦相似度建模用户画像
- 主题模型（如 LDA）提取兴趣主题
- 使用语言模型生成文献的向量表示（embedding），进行匹配
- 利用 BERT / SciBERT / SPECTER 模型编码文献内容

## 代表文献
- Cohan, A., et al. "SPECTER: Document-level Representation Learning using Citation-informed Transformers." ACL 2020.
- Beltagy, I., Lo, K., & Cohan, A. "SciBERT: A Pretrained Language Model for Scientific Text." EMNLP 2019.
