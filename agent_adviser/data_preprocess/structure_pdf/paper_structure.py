import fitz  # PyMuPDF
import re
import sys
import json
from openai import OpenAI
import os
from dotenv_vault import load_dotenv  # pip install --upgrade python-dotenv-vault
load_dotenv()  # https://vault.dotenv.org/ui/ui1
sys.path.append('./')
from agent_adviser.configs.model_config import (DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, AUTHORS_INFO_PROMPT,
                                                ABSTRACT_PROMPT, REFERENCES_PROMPT, APPENDIX_PROMPT)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")


def extract_date_and_title(file_name):
    # 提取日期 (匹配 YYYY 或 YYYY.MM 或 YYYY.MM.DD)
    date_pattern = r"\d{4}(?:\.\d{2})?(?:\.\d{2})?"  # 匹配 YYYY 或 YYYY.MM 或 YYYY.MM.DD
    date_match = re.search(date_pattern, file_name)
    date = date_match.group() if date_match else "日期未找到"

    # 提取论文标题，不使用 look-behind
    title_start = file_name.find(date) + len(date) + 1  # 日期后的一个字符空格
    title = file_name[title_start:].replace(".pdf", "").strip()  # 去掉".pdf"并去除两端空格

    return date, title


def get_authors(text: str):
    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
    )

    completion = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system",
             "content": "你是一个文本处理专家，基于给定的文本信息提取关键信息。"},
            {"role": "user", "content": AUTHORS_INFO_PROMPT.format(TEXT=text)}
        ],
        response_format={
            'type': 'json_object'
        },
        max_tokens=4096
    )
    result = completion.choices[0].message.content

    # 使用 json.loads 将字符串转换为 Python 字典
    data_dict = json.loads(result)

    return data_dict


def get_abstract(text: str):
    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
    )

    completion = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system",
             "content": "你是一个文本处理专家，基于给定的文本信息提取关键信息。"},
            {"role": "user", "content": ABSTRACT_PROMPT.format(TEXT=text)}
        ],
        max_tokens=4096
    )
    result = completion.choices[0].message.content

    return result


def find_and_concatenate(lst, keyword='REFERENCES\n'):
    # 遍历列表，找到 REFERENCES 位置，然后将 REFERENCES 所在的元素后面的页都拼接起来。方便后面提取 REFERENCES 和 附录。
    for i, element in enumerate(lst):
        # 检查元素中是否包含指定的关键词（不区分大小写）
        if keyword.lower() in element.lower():
            # 拼接包含关键词的元素及其后面的所有元素
            result = ' '.join(lst[i:])
            return result
    return ""


def load_partial_json(result):
    # 参考文献太长了，大模型输出的token有限，这时候JSON可能不完整，这个函数就是处理不完整的JSON，将后面的忽略掉，形成完成的JSON，避免出错。
    try:
        # 直接尝试解析完整的 JSON 数据
        data_dict = json.loads(result)
        return data_dict
    except json.JSONDecodeError:
        # 如果解析失败，假设最后一个元素不完整，使用正则去掉最后一个不完整的元素
        # 找到 "references" 列表的部分，移除不完整的最后一个元素
        match = re.search(r'("references": \[.*?)(,[^,]*?)?\s*$', result, re.DOTALL)
        if match:
            # 保留到倒数第二个元素的部分
            corrected_result = "{" + match.group(1) + "\"]}"
            try:
                # 再次尝试解析修复后的 JSON
                data_dict = json.loads(corrected_result)
                return data_dict
            except json.JSONDecodeError as e:
                print(f"仍然无法解析 JSON：{e}")
                return {}
        else:
            print("找不到 'references' 列表")
            return {}


def get_references(text: str):
    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
    )

    completion = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system",
             "content": "你是一个文本处理专家，基于给定的文本信息提取关键信息。"},
            {"role": "user", "content": REFERENCES_PROMPT.format(TEXT=text)}
        ],
        response_format={
            'type': 'json_object'
        },
        max_tokens=4096
    )
    result = completion.choices[0].message.content

    # print(f'-----result------: {result}')

    # 使用 json.loads 将字符串转换为 Python 字典
    data_dict = load_partial_json(result)

    return data_dict


def get_appendix(text: str):
    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
    )

    completion = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system",
             "content": "你是一个文本处理专家，基于给定的文本信息提取关键信息。"},
            {"role": "user", "content": APPENDIX_PROMPT.format(TEXT=text)}
        ],
        max_tokens=4096
    )
    result = completion.choices[0].message.content

    return result


def extract_between_strings_case_insensitive(A, a, b):
    # 将字符串和子串都转换为小写
    A_lower = A.lower()
    a_lower = a.lower()
    b_lower = b.lower()

    # 查找两个子串的位置
    start = A_lower.find(a_lower)
    end = A_lower.find(b_lower, start + len(a_lower))

    if start == -1 or end == -1:
        return ""

    # 返回原始字符串中两个子串之间的部分
    return A[start + len(a):end]


def extract_after_strings_case_insensitive(A, a):
    # 将字符串和子串都转换为小写
    A_lower = A.lower()
    a_lower = a.lower()

    # 查找两个子串的位置
    start = A_lower.find(a_lower)

    if start == -1:
        return ""

    # 返回原始字符串中两个子串之间的部分
    return A[start + len(a):]


def extract_pdf_to_json(pdf_path):
    date, title = extract_date_and_title(pdf_path)

    structured_data = {
        "title": title,
        "authors": None,
        "date": date,
        "abstract": None,
        "body": None,
        "references": None,
        "appendix": None
    }

    document = fitz.open(pdf_path)

    page_0_text = document.load_page(0).get_text("text")

    authors = get_authors(page_0_text)
    abstract = get_abstract(page_0_text)

    structured_data['authors'] = authors
    structured_data['abstract'] = abstract

    content = []

    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text = page.get_text("text")
        content.append(text)

    concatenated_string = find_and_concatenate(content)
    # print(f'-----concatenated_string------: {concatenated_string}')
    references = get_references(concatenated_string)
    # print(f'-----references------: {references}')
    appendix = ""
    if references:
        if 'references' in references:
            references = references['references']  # deepseek输出的references 可能是 {'references': [refs_1, refs_2, ...]}
            # 或者 [refs_1, refs_2, ...]
            if references:  # 可能出现  {'references': []} ，因此需要特殊处理
                append_start = references[-1][-10:]
                # appendix = get_appendix(concatenated_string)  # deepseek由于输出token最大4096，有可能附录比较长，不完整
                appendix = extract_after_strings_case_insensitive(concatenated_string, append_start)

    structured_data['references'] = references
    structured_data['appendix'] = appendix

    full_content = ' '.join(content)
    # save_text_to_file(full_content, file_name="./output/extracted_text.txt")

    body = extract_between_strings_case_insensitive(full_content, abstract[-10:], '\nREFERENCES\n')
    structured_data['body'] = body

    return structured_data


def save_text_to_file(text, file_name="output.txt"):
    with open(file_name, "w", encoding="utf-8") as file:
        file.write(text)
    print(f"Text successfully written to {file_name}")


def main():
    pdf_path = ("./agent_adviser/data/papers/思考&规划/任务规划/2023.09.27 SayPlan- Grounding Large Language Models using 3D "
                "Scene Graphs for Scalable Robot Task Planning.pdf")

    info = extract_pdf_to_json(pdf_path)

    pretty_result = json.dumps(info, indent=4, ensure_ascii=False)
    # 打印格式化后的 JSON
    print(pretty_result)


if __name__ == "__main__":
    main()
