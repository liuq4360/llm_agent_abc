import os
import PyPDF2
from tqdm import tqdm
import sys
sys.path.append('./')
from openai import OpenAI
import os
from dotenv_vault import load_dotenv  # pip install --upgrade python-dotenv-vault
load_dotenv()  # https://vault.dotenv.org/ui/ui1

from automated_article.configs.model_config import DEEPSEEK_MODEL, DEEPSEEK_BASE_URL

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
)


# 子函数1：生成标题
def generate_title(article_content):
    """
    使用大模型生成吸引眼球的中文标题。

    :param article_content: 英文的文章内容
    :return: 生成的中文标题
    """
    prompt = (
        "你是一名专业的编辑，擅长为长篇文章撰写吸引人的中文标题。"
        "标题需要满足以下要求：\n"
        "1. 最好是疑问句，或者包含数字，或者有对比性。\n"
        "2. 用中文撰写，标题需要生动、有趣、简洁。\n"
        "3. 标题长度为10-25个字之间。\n"
        "4. 你只需要生成一个标题。\n"
        "\n"
        "以下是文章内容摘要：\n"
        f"{article_content[:3000]}\n"  # 提供前3000字符供模型参考
        "\n请基于文章内容撰写一个中文标题。"
    )
    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": "你是一名专业的中文编辑。"},
            {"role": "user", "content": prompt},
        ]
    )
    message_content = response.choices[0].message.content.strip()
    return message_content


# 子函数2：生成长文总结
def generate_summary(title, article_content):
    """
    使用 OpenAI 的 GPT 生成文章的中文长文总结（2000-5000字）。

    :param article_content: 英文的文章内容
    :return: 中文长文总结
    """
    prompt = (
        "你是一名专业的技术作家，擅长将英文文章总结为中文长文。"
        "请将以下文章总结成一篇2000-5000字的中文文章，重点突出文章的核心贡献、创新方法及主要结论，并给出数据支撑（如果有的话），但不要提供参考文献。\n"
        "你的总结要确保语言生动有趣，不要太学术，适合大众阅读。\n"
        f"总结的标题为：{title}，请用这个标题，别自己起标题。\n"
        f"确保输出中的标题为markdown的二级标题，即标题以 ## 开头，其他子标题依次递进，分别用三级、四级、五级等标题样式。\n"
        f"给你提供的文章内容是：{article_content[:10000]}\n"  # 提供前10000字符供模型参考
        "\n现在请你撰写总结。"
    )
    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": "你是一名专业的中文技术作家。"},
            {"role": "user", "content": prompt},
        ],
        max_tokens=4096
    )
    message_content = response.choices[0].message.content.strip()
    return message_content


# 主函数：读取 PDF 并生成标题和总结
def process_papers(file_info, file_dir: str = "./automated_article/output/articles/",
                   output_dir: str = "./automated_article/output/summaries"):
    """
    遍历指定目录中的所有 PDF 文件，生成标题和总结，并保存为文本文件。

    :param file_info: [{"file": xx.pdf, "link": xx}]
    :param file_dir:
    :param output_dir:
    """

    os.makedirs(output_dir, exist_ok=True)

    summary_list = []
    for info_ in tqdm(file_info, desc="Processing papers"):
        file = info_['file']
        link = info_['link']
        if not file.endswith(".pdf"):
            continue

        # 读取 PDF 内容
        with open(os.path.join(file_dir, file), "rb") as f:
            reader = PyPDF2.PdfReader(f)
            article_content = "\n".join(page.extract_text() for page in reader.pages)

        # 生成标题
        title = generate_title(article_content)

        # 生成总结
        summary = generate_summary(title, article_content)

        # 保存为文本文件
        output_path = os.path.join(output_dir, f"{title}.md")
        with open(output_path, "w", encoding="utf-8") as out_file:
            out_file.write(f"{summary}\n")

        summary_list.append({
            "title": title,
            "summary": summary,
            "link": link
        })
    return summary_list


def write_abstract_api(file_info, file_dir: str = "./automated_article/output/articles/",
                       output_dir: str = "./automated_article/output/summaries"):
    summary_list = process_papers(file_info, file_dir, output_dir)
    return summary_list
