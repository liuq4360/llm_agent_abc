import sys
import json
from tqdm import tqdm
sys.path.append('./')
from openai import OpenAI
import os
from dotenv_vault import load_dotenv  # pip install --upgrade python-dotenv-vault
load_dotenv()  # https://vault.dotenv.org/ui/ui1
from pdf_translate.configs.model_config import DEEPSEEK_MODEL, DEEPSEEK_BASE_URL

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
)


# 子函数1：生成标题
def generate_en_2_zh_translate(content):
    """
    使用 OpenAI 的 GPT 生成吸引眼球的中文标题。

    :param content: 英文的文章内容
    :return: 生成的中文标题
    """
    prompt = (
        "你是一名专业的翻译，擅长将英文文档翻译成中文。"
        "以下是给你翻译的英文原文：\n"
        f"{content}\n"
        "\n请将上面英文翻译成语句通顺、结构合理、符合中国人表达习惯的中文。注意，如果是阿拉伯数字请不用翻译为中文，英文名、链接等也不用翻译。"
    )
    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": "你是一名专业的英文翻译，你擅长将英文翻译为中文。"},
            {"role": "user", "content": prompt},
        ]
    )
    message_content = response.choices[0].message.content.strip()
    return message_content


def translate_all_pages(text_with_layout, output_path):
    """
    主函数：翻译整个文档。
    """
    # 1. 翻译每一页的文字
    translated_data = []
    for page_data in tqdm(text_with_layout, total=len(text_with_layout), desc=f"translate pages", unit="pages"):
        page_translated = []
        for paragraph in tqdm(page_data, total=len(page_data), desc=f"translate paragraphs", unit="paragraph"):
            original_text = paragraph["text"]
            translated_text = generate_en_2_zh_translate(original_text)
            paragraph["text"] = translated_text
            page_translated.append(paragraph)
        translated_data.append(page_translated)

    # 打开文件（如果文件不存在，将自动创建）
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(json.dumps(translated_data, indent=4, ensure_ascii=False))
