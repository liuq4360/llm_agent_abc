import sys
import json
import argparse
sys.path.append('./')
from sentence_transformers import SentenceTransformer
from agent_adviser.utils.results_cache import get_json_from_redis
from agent_adviser.configs.model_config import SIMILARITY_THRESHOLD

model = SentenceTransformer("all-MiniLM-L6-v2")


def find_max_and_index(values):
    """
    找到列表中最大值及其索引
    :param values: 输入的列表或一维 numpy 数组
    :return: 最大值及其索引 (max_value, index)
    """
    max_value = max(values)
    index = values.index(max_value) if isinstance(values, list) else int(values.argmax())
    return max_value, index


def find_most_similar(text: str, text_list: list):
    """
    找到与目标向量最相似的向量及其相似度
    :param text_list:
    :param text:
    :return: 最相似向量的索引，相似向量，相似度
    """

    embeddings = model.encode([text] + text_list, normalize_embeddings=True)

    scores = embeddings[0] @ embeddings[1:].T
    # print(f"Similarity scores:\n{scores}")

    max_value, index = find_max_and_index(scores)
    return index, text_list[index], max_value


def faq_recall(search_text: str) -> str:
    cache_dict = get_json_from_redis()
    index, text, score = find_most_similar(str(search_text), list(cache_dict.keys()))
    if score > SIMILARITY_THRESHOLD:  # 当相似等分大于此值时，认为这个问题已经被回答过了，可以直接从缓存中返回结果了。
        response_content = cache_dict[text]
        return response_content
    else:
        return ""


if __name__ == '__main__':
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="faq recall api")

    # 添加命令行参数
    parser.add_argument('--text', type=str, required=True, help='搜索词')

    # 解析命令行参数
    args = parser.parse_args()

    text = args.text

    print("----------------------------------------------")
    result = faq_recall(text)
    # 使用 json.dumps 方法来格式化 JSON 数据
    pretty_result = json.dumps(result, indent=4, ensure_ascii=False)
    # 打印格式化后的 JSON
    print(pretty_result)

