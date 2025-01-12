import os
import hashlib
import numpy as np
from sentence_transformers import SentenceTransformer

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


def get_md5(input_string):
    # 创建 MD5 对象
    md5_obj = hashlib.md5()

    # 将字符串编码为字节流，并更新 MD5 对象
    md5_obj.update(input_string.encode('utf-8'))

    # 获取 MD5 值并转换为十六进制字符串
    return md5_obj.hexdigest()


def list_files_with_os(path):
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if not file.startswith('.'):  # 过滤掉隐藏文件
                file_list.append(os.path.join(root, file))
    return file_list


if __name__ == '__main__':
    # all_files = list_files_with_os("./data/papers/")
    # print(len(all_files))
    # print(all_files)

    # 示例向量
    index, text, max_value = find_most_similar("1+1", ["1+1=?", "1+1=2."])
    print(f"index = {index}")
    print(f"text={text}")
    print(f"max_value={max_value}")
