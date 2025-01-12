import sys
import argparse
import json
from typing import List, Dict
sys.path.append('./')
from agent_adviser.configs.model_config import MODEL_PATH
from FlagEmbedding import FlagReranker
from agent_adviser.serving.recall.full_text_search_api import full_search
from agent_adviser.serving.recall.vector_search_api import vector_search

MODEL_LIST = MODEL_PATH["reranker"]


def rerank_search_results(search_text: str, full_search_recall: List[Dict], vec_recall: List[Dict], keep_num: int = 10) -> List[Dict]:
    """
    根据召回结果进行排序
    :param keep_num:
    :param text: 查询词
    :param full_search_recall: 全文检索召回结果
    :param vec_recall: 向量召回结果
    :return:
    """

    full_search_recall_info = []
    for e in full_search_recall:
        e['method'] = 'full_search'
        full_search_recall_info.append(e)

    vec_recall_info = []
    for e in vec_recall:
        e['method'] = 'vector_search'
        vec_recall_info.append(e)

    # 将召回结果合并为一个列表
    recalls = full_search_recall_info + vec_recall_info

    # 去重操作
    unique_items = {}
    for item in recalls:
        item_id = item['_source']['item_id']
        if item_id not in unique_items:
            unique_items[item_id] = item

    # 提取去重后的列表
    recalls = list(unique_items.values())

    pairs = []
    for i in range(len(recalls)):
        recall = recalls[i]
        title = recall["_source"]["title"]
        authors = recall["_source"]["authors"]
        abstract = recall["_source"]["abstract"]
        body = recall["_source"]["body"]

        doc_info = {
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "body": body,
        }
        pairs.append([search_text, str(doc_info)])

    # 召回结果排序
    reranker = FlagReranker(MODEL_LIST["bge-reranker-v2-m3"], use_fp16=True)  # Setting use_fp16 to True speeds up

    if pairs:
        scores = reranker.compute_score(pairs, normalize=True)

        # 合并两个列表为一个由元组构成的列表
        combined_list = list(zip(recalls, scores))

        # 按照元组的第二个元素降序排列
        sorted_combined_list = sorted(combined_list, key=lambda x: x[1], reverse=True)

        # 拆开排列后的列表，保留由元组第一个元素组成的列表
        result_list = []
        for x in sorted_combined_list:
            temp = x[0]
            temp['rerank_score'] = x[1]
            result_list.append(temp)
        return result_list[:keep_num]
    else:
        return []


if __name__ == '__main__':
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="rerank api")

    # 添加命令行参数
    parser.add_argument('--text', type=str, required=True, help='搜索词')

    # 解析命令行参数
    args = parser.parse_args()

    text = args.text

    full_search_recall = full_search(text)
    vec_recall = vector_search(text)

    rerank_results = rerank_search_results(text, full_search_recall, vec_recall, keep_num=10)

    # 使用 json.dumps 方法来格式化 JSON 数据
    pretty_result = json.dumps(rerank_results, indent=4, ensure_ascii=False)
    # 打印格式化后的 JSON
    print(pretty_result)
