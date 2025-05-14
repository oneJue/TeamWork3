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
    corpus = choose_corpus(corpus)

    logger.info("Retrieving Arxiv papers...")
    papers = get_arxiv_paper(args.arxiv_query, args.debug,max_results=args.max_paper_num)
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

