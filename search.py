import json
import time
import re
from loguru import logger

def generate_search_keywords(corpus:list[dict], llm_instance=None) -> list[str]:
    """
    使用LLM分析用户最近阅读的文献，生成检索关键词
    
    Args:
        corpus: 筛选后的文献语料库
        llm_instance: LLM实例，如果为None则使用全局实例
        
    Returns:
        生成的关键词列表
    """
    logger.info("正在使用LLM生成检索关键词...")
    
    # 获取LLM实例
    if llm_instance is None:
        from llm import get_llm
        llm_instance = get_llm()
    
    # 如果语料库为空，返回默认关键词
    if not corpus:
        default_keywords = ["time series", "machine learning", "deep learning", 
                           "neural networks", "data mining", "predictive analytics"]
        logger.warning("语料库为空，使用默认关键词")
        return default_keywords
    
    # 准备提示词
    recent_papers = "\n\n".join([
        f"Title: {paper['data']['title']}\nAbstract: {paper['abstractNote'][:500]}..."  # 限制摘要长度
        for paper in corpus[:10]  # 只使用最近的10篇文章
    ])
    
    prompt = """
As an academic search expert, I need you to analyze the following recently read academic papers and extract keywords and key phrases that represent my research interests. 
These keywords will be used to construct arXiv search queries to find the latest related papers.

Below are the papers I've recently read:
{papers}

Please extract 10-15 keywords and phrases that best represent my research interests. The keywords should:
1. Include domain-specific terms, method names, research objectives, etc.
2. Include both broad field keywords and specific technical terms
3. Be sorted by importance

IMPORTANT: Your response must STRICTLY follow this JSON format:
{{
  "keywords": [
    "keyword1",
    "keyword2",
    "keyword3",
    ...
  ]
}}

DO NOT include any explanations, notes, or additional text outside of this JSON structure.
""".format(papers=recent_papers)

    # 最大重试次数
    max_retries = 3
    retry_count = 0
    default_keywords = ["time series", "machine learning", "deep learning", 
                       "neural networks", "data mining", "predictive analytics"]
    
    while retry_count < max_retries:
        try:
            # 使用llm.py中的接口调用LLM
            messages = [{"role": "user", "content": prompt}]
            response_text = llm_instance.generate(messages)
            
            # 处理回复
            try:
                # 尝试解析JSON响应
                response_json = json.loads(response_text)
                keywords = response_json.get("keywords", [])
                
                # 验证关键词列表
                if not keywords or not isinstance(keywords, list):
                    raise ValueError("关键词列表为空或格式不正确")
                
                logger.info(f"成功生成了{len(keywords)}个检索关键词")
                logger.debug(f"生成的关键词: {', '.join(keywords[:5])}...")
                return keywords
                
            except json.JSONDecodeError:
                # 如果无法解析JSON，尝试使用正则表达式提取关键词
                logger.warning("无法解析LLM返回的JSON，尝试使用正则表达式提取关键词")
                import re
                # 尝试提取按行分隔的关键词列表
                lines = [line.strip() for line in response_text.split('\n') if line.strip()]
                
                # 去除可能的编号和其他标记
                keywords = []
                for line in lines:
                    # 去除可能的编号、引号、破折号等
                    cleaned = re.sub(r'^[\d\.\"\'\-\*]+\s*', '', line).strip()
                    if cleaned:
                        keywords.append(cleaned)
                
                if keywords:
                    logger.info(f"使用正则表达式提取了{len(keywords)}个关键词")
                    return keywords
                
                raise ValueError("无法从响应中提取关键词")
                
        except Exception as e:
            retry_count += 1
            logger.warning(f"第{retry_count}次尝试生成关键词失败: {e}")
            if retry_count >= max_retries:
                logger.error(f"达到最大重试次数({max_retries})，使用默认关键词")
                return default_keywords
            
            # 短暂等待后重试
            time.sleep(2)
    
    # 如果所有尝试都失败，返回默认关键词
    return default_keywords

def build_arxiv_query(keywords:list[str], max_terms:int=8) -> str:
    """
    基于关键词构建arXiv检索式
    
    Args:
        keywords: 关键词列表
        max_terms: 最大使用的关键词数量
        
    Returns:
        arXiv检索式
    """
    # 确保有关键词可用
    if not keywords:
        default_query = "ti:\"machine learning\" OR ti:\"deep learning\" OR ti:\"time series\""
        logger.warning(f"关键词列表为空，使用默认检索式: {default_query}")
        return default_query
    
    # 选择最重要的几个关键词（假设按重要性排序）
    selected_keywords = keywords[:min(len(keywords), max_terms)]
    
    # 构建高级搜索查询
    try:
        # 标题或摘要中包含关键词，使用OR连接
        query_parts = []
        
        # 添加标题和摘要搜索
        for keyword in selected_keywords:
            # 确保关键词是字符串
            if not isinstance(keyword, str) or not keyword.strip():
                continue
                
            # 如果关键词包含空格，添加引号
            if ' ' in keyword:
                keyword = f'"{keyword}"'
            query_parts.append(f"ti:{keyword} OR abs:{keyword}")
        
        # 确保有有效的查询部分
        if not query_parts:
            default_query = "ti:\"machine learning\" OR ti:\"deep learning\" OR ti:\"time series\""
            logger.warning(f"无法创建有效的查询部分，使用默认检索式: {default_query}")
            return default_query
            
        # 连接所有查询部分
        query = " OR ".join(query_parts)
        
        logger.info(f"构建的arXiv检索式: {query}")
        return query
        
    except Exception as e:
        default_query = "ti:\"machine learning\" OR ti:\"deep learning\" OR ti:\"time series\""
        logger.error(f"构建检索式时出错: {e}，使用默认检索式: {default_query}")
        return default_query
