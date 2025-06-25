from paper import ArxivPaper
import math
from tqdm import tqdm
from email.header import Header
from email.mime.text import MIMEText
from email.utils import parseaddr, formataddr
import smtplib
import datetime
from loguru import logger
import json
import base64
from typing import Optional, List

"""
    说明：经过修改后，ArxivPaper类需要

    正式使用时，需要修改 render_email() 方法中的测试代码

    main 中需要调用 render_sidebar() 方法，提供专业领域推荐和文章推荐的链接。
    不提供也没关系
"""

# # -----------------------------------------------------------
# # 测试代码
# class Author:
#         def __init__(self, name):
#             self.name = name

# class ArxivPaper:
#     def __init__(self, title=None, authors=None, tldr=None, 
#                     pdf_url=None, code_url=None, affiliations=None, 
#                     date=None, comment=None, abs_url=None, score=None, summary=None, 
#                     tex=None, labels=None):
        
#         self.title = title
#         self.abs_url = abs_url
#         self.comment = comment
#         self.date = date
#         self.summary = summary
#         self.authors = authors
#         self.pdf_url = pdf_url
#         self.code_url = code_url
#         self.tex = tex
#         self.tldr = tldr
#         self.affiliations = affiliations
    
#         self.score = score
#         self.labels  = labels

# def test_render_email():
#     papers = [
#         ArxivPaper(
#             title="Test Paper 1",
#             authors=[Author("Alice"), Author("Bob")],
#             summary="This is a test abstract.",
#             pdf_url="https://example.com/paper1.pdf",
#             code_url="https://github.com/test/repo1",
#             affiliations=["University A", "University B"],
#             date="2023-10-01",
#             comment="Good paper!",
#             abs_url="https://arxiv.org/abs/1234.5678",
#             score=6.9,
#             labels=["test_label 1","test_label 2"],
#         ),
#         ArxivPaper(
#             title="Test Paper 2",
#             authors=[Author("Charlie")],
#             summary="Another test abstract.",
#             pdf_url="https://example.com/paper2.pdf",
#             affiliations=["University C"],
#             date="2023-09-30",
#             abs_url="https://arxiv.org/abs/9876.5432",
#             score=10.0,
#             labels=["test_label 3"]
#         )
#     ]
#     field = "https://example.com/fiedl-recommendation.html"
#     article = "https://example.com/article.html"

#     # 调用 render_email 方法生成 HTML
#     html_content = render_email(papers,field_recommend=field,article_recommend=article)

#     send_email(sender="1812291127@qq.com", receiver='["51275903100@stu.ecnu.edu.cn"]', 
#                password="xdoimelilwcxdecb", smtp_server="smtp.qq.com", smtp_port=465,
#                html=html_content)

#     # 将 HTML 写入临时文件并用浏览器打开
#     with open("test_email.html", "w", encoding="utf-8") as f:
#         f.write(html_content)

#     import webbrowser
#     webbrowser.open("test_email.html")
# # -----------------------------------------------------------
import hashlib

class StringColorMapper:
    """
    一个将字符串映射到预定义颜色库的工具类。
    """
    def __init__(self,):
       
        self.palette = ['#ea1e63','#01bdd6','#b1e7fb','#4db151','#8cc24a','#cddc39','#ffeb3c','#ffc10a','#ff9803',
                        '#fe5722','#6fc0b1','#b9e479','#fecdae','#f0f5de','#fee7d7','#c1fdfd']
        self.palette_size = len(self.palette)

    def get_color(self, input_string: str) -> str:
        """
        为一个给定的字符串获取其对应的颜色。
        """
        encoded_string = input_string.encode('utf-8')
        hash_object = hashlib.sha256(encoded_string)
        hex_digest = hash_object.hexdigest()
        hash_int = int(hex_digest, 16)
        index = hash_int % self.palette_size
        return self.palette[index]
    

with open("webpages/logo.png", "rb") as image_file:
    logo_base64 = base64.b64encode(image_file.read()).decode()


logo_data_uri = f'data:image/png;base64,{logo_base64}'


with open("webpages/framework.html", "r", encoding="utf-8") as f:
    framework = f.read().replace('{logo}', logo_data_uri)

td_style : str = "font-size: 14px; color: #333; padding: 8px 0;"


# def render_sidebar(field_recommend: str = "#", article_recommend: str = "#") -> None:
#     global framework
#     framework = framework.replace('{field_recommend}', field_recommend)
#     framework = framework.replace('{article_recommend}', article_recommend)


def get_empty_html() -> str:
    with open("webpages/block_template_empty.html", "r", encoding="utf-8") as f:
        block_template = f.read()
    return block_template


def get_block_html(
    title: str, 
    authors: str, 
    rate: str, 
    abstract: str, 
    pdf_url: str,
    code_url: Optional[str] = None, 
    affiliations: Optional[str] = None, 
    date: Optional[str] = None, 
    comment: Optional[str] = None,
    abs_url: Optional[str] = None, 
    labels: Optional[List[str]] = None,
    color_map:  Optional[StringColorMapper] = None
) -> str:
    with open("webpages/block_template.html", "r", encoding="utf-8") as f:
        block_template = f.read()

    arxiv_id = get_arxiv_id(abs_url)

    code = f'<a href="{code_url}" class="code">Code</a>' if code_url else ''

    comments = f'<tr><td style="{td_style}"><strong>Comment:</strong> {comment}</td></tr>' if comment else ''

    if labels and isinstance(labels, list) and len(labels) > 0:
        label_html = f'<tr><td style="{td_style}"><strong>Label: </strong>'
        for label in labels:
            label_html += f'<span class="label" style="background-color:{color_map.get_color(label)}">{label}</span>'
        label_html += '</td></tr>'
    else:
        label_html = ''

    return block_template.format(
        title=title,
        authors=authors,
        rate=rate,
        arxiv_id=arxiv_id,
        abs_url=abs_url,
        abstract=abstract,
        pdf_url=pdf_url,
        code=code,
        affiliations=affiliations,
        date=date,
        comment=comments,
        labels=label_html
    )


def get_arxiv_id(abs_url: str) -> str:
    if abs_url and "arxiv.org/abs" in abs_url:
        return abs_url.split('/')[-1]
    return ""

def get_stars(score: float) -> str:
    full_star = '<span class="full-star">⭐</span>'
    half_star = '<span class="half-star">⭐</span>'
    low = 6
    high = 8
    if score <= low:
        return ''
    elif score >= high:
        return full_star * 5
    else:
        interval = (high - low) / 10
        star_num = math.ceil((score - low) / interval)
        full_star_num = int(star_num / 2)
        half_star_num = star_num - full_star_num * 2
        return '<div class="star-wrapper">' + full_star * full_star_num + half_star * half_star_num + '</div>'


def render_email(papers: list[ArxivPaper],field_recommend:str="#",article_recommend:str="#",papers_coarse=None) -> str:
    parts = []
    if len(papers) == 0:
        return framework.replace('__CONTENT__', get_empty_html())

    # render_sidebar(field_recommend=field_recommend,article_recommend=article_recommend)
    papers = sorted(papers, key=lambda p: p.date, reverse=True)

    color_map = StringColorMapper()
    for p in tqdm(papers, desc='Rendering Email'):
        # rate = get_stars(p.score)
        rate = '<span class="full-star">⭐</span>' * 5  # 测试用
        labels = p.labels
        authors = [a.name for a in p.authors[:5]]
        authors = ', '.join(authors)
        if len(p.authors) > 5:
            authors += ', ...'
        if p.affiliations is not None:
            affiliations = p.affiliations[:5]
            affiliations = ', '.join(affiliations)
            if len(p.affiliations) > 5:
                affiliations += ', ...'
        else:
            affiliations = 'Unknown Affiliation'
        # parts.append(get_block_html(title=p.title, authors=authors, rate=rate, abstract=p.summary, 
        #                             pdf_url=p.pdf_url, code_url=p.code_url, affiliations=affiliations,
        #                             date=p.date, comment=p.comment, abs_url=p.abs_url,labels=p.labels))
        parts.append(get_block_html(title=p.title, authors=authors, rate=rate, abstract=p.summary, 
                                    pdf_url=p.pdf_url, code_url=p.code_url, affiliations=affiliations,
                                    date=p.date, comment=p.comment, abs_url=p.abs_url,labels=labels,color_map=color_map))

    content = '<br>' + '</br><br>'.join(parts) + '</br>'

    coarse_parts=[]
    papers_coarse = sorted(papers_coarse, key=lambda p: p.date, reverse=True)
    for p in tqdm(papers_coarse, desc='Rendering Email'):
        # rate = get_stars(p.score)
        rate = '<span class="full-star">⭐</span>' * 5  # 测试用
        labels = p.labels
        authors = [a.name for a in p.authors[:5]]
        authors = ', '.join(authors)
        if len(p.authors) > 5:
            authors += ', ...'
        if p.affiliations is not None:
            affiliations = p.affiliations[:5]
            affiliations = ', '.join(affiliations)
            if len(p.affiliations) > 5:
                affiliations += ', ...'
        else:
            affiliations = 'Unknown Affiliation'
        coarse_parts.append(get_block_html(title=p.title, authors=authors, rate=rate, abstract=p.summary, 
                                    pdf_url=p.pdf_url, code_url=p.code_url, affiliations=affiliations,
                                    date=p.date, comment=p.comment, abs_url=p.abs_url,labels=labels,color_map=color_map))
        coarse_content = '<br>' + '</br><br>'.join(coarse_parts) + '</br>'
    return framework.replace('__CONTENT__', content).replace('__CONTENT_FIELD__', coarse_content)


def send_email(sender: str, receiver: str, password: str, smtp_server: str, smtp_port: int, html: str) -> None:
    def _format_addr(s):
        name, addr = parseaddr(s)
        return formataddr((Header(name, 'utf-8').encode(), addr))

    print(receiver)
    logger.warning(f"receiver")

    try:
        receivers = receiver
    except json.JSONDecodeError:
        logger.error("Invalid receiver format, must be a JSON list.")
        raise

    msg = MIMEText(html, 'html', 'utf-8')
    msg['From'] = _format_addr('Github Action <%s>' % sender)
    # msg['To'] = _format_addr('You <%s>' % receiver)
    msg['To'] = ', '.join([_format_addr(f'<%s>' % receiver) for receiver in receivers])
    today = datetime.datetime.now().strftime('%Y/%m/%d')
    msg['Subject'] = Header(f'Daily arXiv {today}', 'utf-8').encode()

    try:
        # 修改：直接使用SMTP_SSL
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
    except Exception as e:
        logger.warning(f"Failed to use TLS. {e}")
        logger.warning(f"Try to use SSL.")
        server = smtplib.SMTP_SSL(smtp_server, smtp_port)

    server.login(sender, password)
    server.sendmail(sender, receivers, msg.as_string())
    server.quit()


# # 测试代码
# if __name__ == "__main__":
#     test_render_email()