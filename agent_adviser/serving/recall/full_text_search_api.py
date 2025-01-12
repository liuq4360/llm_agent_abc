from elasticsearch import Elasticsearch
import sys
import json
import argparse
from typing import List, Dict
sys.path.append('./')
from agent_adviser.configs.model_config import HOST, PORT, INDEX_NAME, MIN_SEARCH_SCORE


def full_search(search_text: str) -> List[Dict]:
    # 创建Elasticsearch客户端实例
    es = Elasticsearch(
        hosts=[{'host': HOST, 'port': PORT, 'scheme': 'http'}],
    )
    """
        •	query: 搜索的关键词或短语。
        •	fields: 一个包含多个字段名称的列表，指定在哪些字段中搜索。
        •	type: multi_match 查询的类型，支持以下几种：
        •	best_fields: 默认选项，在所有字段中查找最匹配的字段，并返回该字段的最高分数。
        •	most_fields: 在多个字段中查找最匹配的字段，并累加所有匹配字段的分数。
        •	cross_fields: 适合用于那些在不同字段中分布的查询词。
        •	phrase: 进行短语搜索。
        •	phrase_prefix: 用于实现自动完成功能。
        •	operator: or 或 and，定义匹配多个词时的逻辑操作。
    """

    query_condition = {
        "query": {
            "bool": {  # 使用bool查询以组合多个查询条件
                "must": [  # 必须满足的条件
                    {
                        "multi_match": {
                            "query": search_text,
                            "fields": ["title^3", "abstract^2", "body"],
                            "type": "best_fields",
                            "operator": "or"
                        }
                    }
                ]
            }
        }
    }

    response = es.search(
        index=INDEX_NAME,
        body=query_condition,
        # size=keep_num,  # 返回的结果数量限制,
        min_score=MIN_SEARCH_SCORE,  # 设置最小得分
        source=["item_id", "title", "authors", "date", "abstract", "body"]  # 定义返回的字段
    )

    # 处理响应
    if response['hits']['total']['value'] > 0:
        print("Fulltext Search Found %d documents." % response['hits']['total']['value'])
        # for hit in response['hits']['hits']:
        #     print("ID:", hit['_id'])
        #     print("Score:", hit['_score'])
        #     print("Source:", hit['_source'])
        #     print("-" * 40)
        return response['hits']['hits']
    else:
        print("No documents found.")
        return []


if __name__ == '__main__':
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="full text search api")

    # 添加命令行参数
    parser.add_argument('--text', type=str, required=True, help='搜索词')

    # 解析命令行参数
    args = parser.parse_args()

    text = args.text

    print("----------------------------------------------")
    result = full_search(text)
    # 使用 json.dumps 方法来格式化 JSON 数据
    pretty_result = json.dumps(result, indent=4, ensure_ascii=False)
    # 打印格式化后的 JSON
    print(pretty_result)
