import sys
import json
import argparse
from openai import OpenAI
sys.path.append('./')
import os
from dotenv_vault import load_dotenv  # pip install --upgrade python-dotenv-vault
load_dotenv()  # https://vault.dotenv.org/ui/ui1
from agent_adviser.configs.model_config import (ZERO_SAMPLE_CLASSIFICATION_PROMPT,
                                                DEEPSEEK_BASE_URL, DEEPSEEK_MODEL)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
    )


def semantics_classify(text):
    completion = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system",
             "content": "你是一个有帮助的人工智能助手，基于给定的Prompt来回答问题。"},
            {"role": "user", "content": ZERO_SAMPLE_CLASSIFICATION_PROMPT.format(SEARCH_TEXT=text)}
        ],
    )
    result = completion.choices[0].message.content
    return result


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="semantic classify api")

    # 添加命令行参数
    parser.add_argument('--text', type=str, required=True, help='搜索词')

    # 解析命令行参数
    args = parser.parse_args()

    text = args.text

    info = semantics_classify(text)

    # 使用 json.dumps 方法来格式化 JSON 数据
    pretty_json = json.dumps(info, indent=4, ensure_ascii=False)

    # 打印格式化后的 JSON
    print(pretty_json)
