import sys
from fastapi import Query
import argparse
import json
sys.path.append('./')
from agent_adviser.serving.recall.full_text_search_api import full_search
from agent_adviser.serving.recall.vector_search_api import vector_search
from agent_adviser.serving.rerank.search_rerank import rerank_search_results


def search_api(search_text: str = Query(..., description="搜索词"),
               keep_num: int = Query(1000, description="返回结果数量")):
    """
    搜索API
    :param search_text: 用户输入的搜索词
    :param keep_num: 最终返回的文档数量
    :return:
    """
    full_search_recall = full_search(search_text)
    vec_recall = vector_search(search_text)
    rerank_results = rerank_search_results(search_text, full_search_recall, vec_recall, keep_num=keep_num)
    return rerank_results


if __name__ == '__main__':
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="search pipeline api")

    # 添加命令行参数
    parser.add_argument('--text', type=str, required=True, help='搜索词')

    # 解析命令行参数
    args = parser.parse_args()

    text = args.text

    rerank_results = search_api(text, 2)

    # name_list = [x['_source']['title'] for x in rerank_results]

    # 使用 json.dumps 方法来格式化 JSON 数据
    pretty_json = json.dumps(rerank_results, indent=4, ensure_ascii=False)

    # 打印格式化后的 JSON
    print(pretty_json)
