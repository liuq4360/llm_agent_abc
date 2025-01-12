import sys
from openai import OpenAI
sys.path.append('./')
import os
from dotenv_vault import load_dotenv  # pip install --upgrade python-dotenv-vault
load_dotenv()  # https://vault.dotenv.org/ui/ui1
from agent_adviser.configs.model_config import (DEEPSEEK_BASE_URL, DEEPSEEK_MODEL)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
)


# 将对话历史也作为API调用的上下文
def deepseek_chat(history):
    """

    :param history: 下面提供2个案例。
        案例1：
        [('1+1=？', None)]
        案例2：
        [('1+1=？', '1 + 1 = 2。这是基本的加法运算。'), ('写一首5言绝句。', None)]
    :return:
    """
    # 构建对话历史作为上下文
    messages = [{"role": "system", "content": "你是一个有帮助的人工智能助手，基于给定的Prompt来回答问题。"}]
    for human_message, assistant_message in history:
        if human_message:
            messages.append({"role": "user", "content": human_message})
        if assistant_message:
            messages.append({"role": "assistant", "content": assistant_message})

    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=messages,
        stream=True,  # 开启流式响应
    )
    response_content = ""  # 用于拼接完整的响应内容
    for chunk in response:
        if hasattr(chunk, "choices") and chunk.choices[0].delta:  # 确保属性存在
            content = chunk.choices[0].delta.content  # 使用属性访问
            response_content += content  # 累积响应内容
            yield response_content  # 按块返回累积的响应内容


if __name__ == '__main__':
    history = [["1+1=？", "1 + 1 = 2。这是基本的加法运算。"], ["写一首5言绝句。", None]]
    for partial_response in deepseek_chat(history):
        print(partial_response)
