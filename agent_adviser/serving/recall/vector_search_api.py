from elasticsearch import Elasticsearch
import sys
import json
import argparse
from typing import List, Dict
sys.path.append('./')
from agent_adviser.configs.model_config import HOST, PORT, INDEX_NAME, MIN_KNN_SIMILARITY
from agent_adviser.utils.text_embedding import embed_text


def vector_search(search_text: str = "llm agent") -> List[Dict]:
    # 创建Elasticsearch客户端实例
    es = Elasticsearch(
        hosts=[{'host': HOST, 'port': PORT, 'scheme': 'http'}],
    )

    # 定义查询向量
    query_vector = embed_text(search_text)

    search_body = {
        # "size": 10,
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'semantic_vector')",
                    "params": {
                        "query_vector": query_vector
                    }
                }
            }
        },
        "min_score": MIN_KNN_SIMILARITY  # 0.5 + 1.0 因为余弦相似度返回的范围是[-1,1]
    }

    # 执行查询
    try:
        response = es.search(index=INDEX_NAME,
                             body=search_body,
                             source=["item_id", "title", "authors", "date", "abstract", "body"],  # 定义返回的字段
                             size=1000  # 否则默认返回匹配度大于 MIN_KNN_SIMILARITY 的前10个文档。
                             )
        # 处理响应
        if response['hits']['total']['value'] > 0:
            print("Vector Search Found %d similar documents." % response['hits']['total']['value'])
            # for hit in response['hits']['hits']:
            #     print("ID:", hit['_id'])
            #     print("Score:", hit['_score'])
            #     print("Source:", hit['_source'])
            #     print("-" * 40)
            return response['hits']['hits']
        else:
            print("No similar documents found.")
            return []
    except Exception as e:
        print("Error:", e)
        return []


def knn_search(search_text="llm agent"):
    # 创建Elasticsearch客户端实例
    es = Elasticsearch(
        hosts=[{'host': HOST, 'port': PORT, 'scheme': 'http'}],
    )

    # 定义查询向量
    query_vector = embed_text(search_text)

    # 使用 knn_search API 执行向量搜索
    response = es.knn_search(
        index=INDEX_NAME,
        knn={
            "field": "semantic_vector",
            "query_vector": query_vector,
            "k": 10,  # 返回最近的 10 个邻居
            "num_candidates": 100  # 候选数量，用于加速搜索，设置一个合理的值以确保精度和性能
        },
        source=["item_id", "title", "authors", "date", "abstract", "body"]  # 定义返回的字段
    )

    # 处理响应
    if response['hits']['total']['value'] > 0:
        print("Found %d similar documents." % response['hits']['total']['value'])
        # for hit in response['hits']['hits']:
        #     print("ID:", hit['_id'])
        #     print("Score:", hit['_score'])
        #     print("Source:", hit['_source'])
        #     print("-" * 40)
        return response['hits']['hits']
    else:
        print("No similar documents found.")
        return []


if __name__ == '__main__':
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="full text search api")

    # 添加命令行参数
    parser.add_argument('--text', type=str, required=True, help='搜索词')

    # 解析命令行参数
    args = parser.parse_args()

    text = args.text

    vec_recall = vector_search(text)

    # 使用 json.dumps 方法来格式化 JSON 数据
    pretty_result = json.dumps(vec_recall[:5], indent=4, ensure_ascii=False)
    # 打印格式化后的 JSON
    print(pretty_result)
