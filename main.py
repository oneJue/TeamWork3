import arxiv
import argparse
import os
import sys
from dotenv import load_dotenv
load_dotenv(override=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pyzotero import zotero
from recommender import rerank_paper
from construct_email import render_email, send_email
from tqdm import trange,tqdm
from loguru import logger
from gitignore_parser import parse_gitignore
from tempfile import mkstemp
from paper import ArxivPaper
from llm import set_global_llm
import feedparser
import json

def get_zotero_corpus(id:str,key:str) -> list[dict]:
    zot = zotero.Zotero(id, 'user', key)
    collections = zot.everything(zot.collections())
    collections = {c['key']:c for c in collections}
    corpus = zot.everything(zot.items(itemType='conferencePaper || journalArticle || preprint'))
    corpus = [c for c in corpus if c['data']['abstractNote'] != '']
    def get_collection_path(col_key:str) -> str:
        if p := collections[col_key]['data']['parentCollection']:
            return get_collection_path(p) + ' / ' + collections[col_key]['data']['name']
        else:
            return collections[col_key]['data']['name']
    for c in corpus:
        paths = [get_collection_path(col) for col in c['data']['collections']]
        c['paths'] = paths

    print(corpus)
    
    return corpus

# def filter_corpus(corpus:list[dict], pattern:str) -> list[dict]:
#     _,filename = mkstemp()
#     with open(filename,'w') as file:
#         file.write(pattern)
#     matcher = parse_gitignore(filename,base_dir='./')
#     new_corpus = []
#     for c in corpus:
#         match_results = [matcher(p) for p in c['paths']]
#         if not any(match_results):
#             new_corpus.append(c)
#     os.remove(filename)
#     return new_corpus

# 获取标题，下载时间，摘要
def choose_corpus(corpus:list[dict]) -> dict:
    new_corpus = []
    for c in corpus:
        c_dict = {'key':c['key'], 'title':c['data']['title'], 'dateAdded':c['data']['dateAdded'], 'abstractNote':c['data']['abstractNote']}
        new_corpus.append(c_dict)
    return new_corpus

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
        f"Title: {paper['title']}\nAbstract: {paper['abstractNote'][:500]}..."  # 限制摘要长度
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
            import time
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

def get_authors(authors, first_author = False):
    output = str()
    if first_author == False:
        output = ", ".join(str(author) for author in authors)
    else:
        output = authors[0]
    return output
    
def sort_papers(papers):
    output = dict()
    keys = list(papers.keys())
    keys.sort(reverse=True)
    for key in keys:
        output[key] = papers[key]
    return output    

def get_arxiv_paper(query: str, debug: bool = False, max_results: int = 30) -> list[ArxivPaper]:
    # 创建 arxiv 搜索引擎实例
    search_engine = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    papers = []
    for result in search_engine.results():
        # 将论文封装成 ArxivPaper 对象
        paper = ArxivPaper(result)
        papers.append(paper)
        
    return papers



parser = argparse.ArgumentParser(description='Recommender system for academic papers')

def add_argument(*args, **kwargs):
    def get_env(key:str,default=None):
        # handle environment variables generated at Workflow runtime
        # Unset environment variables are passed as '', we should treat them as None
        v = os.environ.get(key)
        if v == '' or v is None:
            return default
        return v
    parser.add_argument(*args, **kwargs)
    arg_full_name = kwargs.get('dest',args[-1][2:])
    env_name = arg_full_name.upper()
    env_value = get_env(env_name)
    if env_value is not None:
        #convert env_value to the specified type
        if kwargs.get('type') == bool:
            env_value = env_value.lower() in ['true','1']
        else:
            env_value = kwargs.get('type')(env_value)
        parser.set_defaults(**{arg_full_name:env_value})


if __name__ == '__main__':
    
    add_argument('--zotero_id', type=str, default='15385713', help='Zotero user ID')
    add_argument('--zotero_key', type=str, default = 'CWVYk463YUtPFKIpda7kqfKH', help='Zotero API key')
    add_argument('--zotero_ignore',type=str,help='Zotero collection to ignore, using gitignore-style pattern.')
    add_argument('--send_empty', type=bool, help='If get no arxiv paper, send empty email',default=False)
    add_argument('--max_paper_num', type=int, help='Maximum number of papers to recommend',default=2)
    add_argument('--arxiv_query', type=str, default='ti:("Time series" OR "Time-series")', help='Arxiv search query')
    add_argument('--smtp_server', type=str,default='smtp.qq.com', help='SMTP server')
    add_argument('--smtp_port', type=int, default='465', help='SMTP port')
    add_argument('--sender', type=str, default='1812291127@qq.com', help='Sender email address')
    add_argument('--receiver', type=str,  default='["51275903106@stu.ecnu.edu.cn"]', help='Receiver email address')
    add_argument('--sender_password', type=str, default='xdoimelilwcxdecb', help='Sender email password')
    add_argument('--use_llm_keywords', type=bool, help='If get no arxiv paper, send empty email',default=False)
    add_argument(
        "--use_llm_api",
        type=bool,
        help="Use OpenAI API to generate TLDR",
        default=True,
    )
    add_argument(
        "--openai_api_key",
        type=str,
        help="OpenAI API key",
        default="sk-37ca05e6889e4d61aa2a08cc1d55339c",
    )
    add_argument(
        "--openai_api_base",
        type=str,
        help="OpenAI API base URL",
        default="https://chat.ecnu.edu.cn/open/api/v1",
    )
    add_argument(
        "--model_name",
        type=str,
        help="LLM Model Name",
        default="ecnu-plus",
    )
    add_argument(
        "--language",
        type=str,
        help="Language of TLDR",
        default="Chinese",
    )
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    args = parser.parse_args()
    assert (
        not args.use_llm_api or args.openai_api_key is not None
    )  # If use_llm_api is True, openai_api_key must be provided
    if args.debug:
        logger.remove()
        logger.add(sys.stdout, level="DEBUG")
        logger.debug("Debug mode is on.")
    else:
        logger.remove()
        logger.add(sys.stdout, level="INFO")

    # starting
    logger.info("Retrieving Zotero corpus...")
    corpus = get_zotero_corpus(args.zotero_id, args.zotero_key)
    logger.info(f"Retrieved {len(corpus)} papers from Zotero.")
    # if args.zotero_ignore:
    #     logger.info(f"Ignoring papers in:\n {args.zotero_ignore}...")
    #     # corpus = filter_corpus(corpus, args.zotero_ignore)
    #     corpus = choose_corpus(corpus)
    #     logger.info(f"Remaining {len(corpus)} papers after filtering.")
    # # ending
    # corpus = choose_corpus(corpus)
    logger.info("Generate Keywords...")
    keywords = generate_search_keywords(corpus)
    query = build_arxiv_query(keywords, args.max_keywords)

    logger.info("Retrieving Arxiv papers...")
    if args.use_llm_keywords:
        papers = get_arxiv_paper(query, args.debug, max_results=args.max_paper_num)
    else:
        papers = get_arxiv_paper(args.arxiv_query, args.debug, max_results=args.max_paper_num)

    if len(papers) == 0:
        logger.info("No new papers found. Yesterday maybe a holiday and no one submit their work :). If this is not the case, please check the ARXIV_QUERY.")
        if not args.send_empty:
          exit(0)
    else:
        logger.info("Reranking papers...")
        papers = rerank_paper(papers, corpus)
        if args.max_paper_num != -1:
            papers = papers[:args.max_paper_num]
        if args.use_llm_api:
            logger.info("Using OpenAI API as global LLM.")
            set_global_llm(api_key=args.openai_api_key, base_url=args.openai_api_base, model=args.model_name, lang=args.language)
        else:
            logger.info("Using Local LLM as global LLM.")
            set_global_llm(lang=args.language)

    html = render_email(papers)
    logger.info("Sending email...")
    send_email(args.sender, args.receiver, args.sender_password, args.smtp_server, args.smtp_port, html)
    logger.success("Email sent successfully! If you don't receive the email, please check the configuration and the junk box.")

