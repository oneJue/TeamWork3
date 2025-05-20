# from llama_cpp import Llama
from openai import OpenAI
from loguru import logger

GLOBAL_LLM = None

class LLM:
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None,lang: str = "English"):
        # if api_key:
        self.llm = OpenAI(api_key=api_key, base_url=base_url)
        # else:
        #     self.llm = Llama.from_pretrained(
        #         repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF",
        #         filename="qwen2.5-3b-instruct-q4_k_m.gguf",
        #         n_ctx=5_000,
        #         n_threads=4,
        #         verbose=False,
        #     )
        self.model = model
        self.lang = lang

    def generate(self, messages: list[dict]) -> str:
        if isinstance(self.llm, OpenAI):
            response = self.llm.chat.completions.create(messages=messages,temperature=0,model=self.model)
            return response.choices[0].message.content
        else:
            response = self.llm.create_chat_completion(messages=messages,temperature=0)
            return response["choices"][0]["message"]["content"]

    def generate_article_tags(self, title: str, abstract: str) -> tuple[str, str]:
        """
        生成文章的类型标签和主题标签

        参数:
            title: 文章标题
            abstract: 文章摘要

        返回:
            包含两个标签的元组 (type_tag, topic_tag)
            type_tag: 文章类型标签，如"研究论文"、"综述"等
            topic_tag: 文章主题标签，包含模型和关键词信息
        """
        prompt = f"""请根据以下学术文章信息生成两个分类标签：

        标题: {title}
        摘要: {abstract}

        要求生成两个标签，用"|"分隔：
        1. 文章类型标签（如：研究论文、综述、案例研究、方法论等）
        2. 文章主题标签（包含：使用的模型/方法 + 2-3个关键词）

        示例：
        "研究论文 | Transformer模型 自然语言处理 文本生成"
        "综述 | 深度学习 医学影像 癌症检测"
        """

        response = self.generate([
            {"role": "system", "content": "你是一个学术文章分类专家"},
            {"role": "user", "content": prompt}
        ])

        # 解析响应
        if "|" in response:
            type_tag, topic_tag = response.split("|", 1)
            return type_tag.strip(), topic_tag.strip()
        return "未知类型", "未知主题"  # 默认值

def set_global_llm(api_key: str = None, base_url: str = None, model: str = None, lang: str = "English"):
    global GLOBAL_LLM
    GLOBAL_LLM = LLM(api_key=api_key, base_url=base_url, model=model, lang=lang)

def get_llm() -> LLM:
    if GLOBAL_LLM is None:
        logger.info("No global LLM found, creating a default one. Use `set_global_llm` to set a custom one.")
        set_global_llm()
    return GLOBAL_LLM