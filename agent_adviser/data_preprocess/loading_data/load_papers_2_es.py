import sys
import json
sys.path.append('./')
from elasticsearch import Elasticsearch, helpers
from agent_adviser.configs.model_config import HOST, PORT, INDEX_NAME
from agent_adviser.data_preprocess.structure_pdf.paper_structure import extract_pdf_to_json
from agent_adviser.utils.little_tools import get_md5, list_files_with_os
from agent_adviser.utils.text_embedding import embed_text


def create_index():
    # 创建一个 Elasticsearch 客户端实例
    es = Elasticsearch(
        hosts=[{'host': HOST, 'port': PORT, 'scheme': 'http'}],
        # http_auth=('username', 'password'),  # 如果需要认证，请添加此行
        # scheme='https',  # 如果使用 HTTPS，取消此行注释
    )

    # 映射配置
    mapping = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        },
        "mappings": {
            "properties": {
                "item_id": {
                    "type": "keyword"
                },
                "title": {
                    "type": "text",
                    "analyzer": "standard"
                },
                "authors": {
                    "type": "nested",
                    "properties": {
                        "name": {"type": "text"},
                        "work": {"type": "text"},
                        "contact": {"type": "text"}
                    }
                },
                "date": {
                    "type": "date",
                    "format": "yyyy-MM-dd"
                },
                "abstract": {
                    "type": "text",
                    "analyzer": "standard"
                },
                "body": {
                    "type": "text",
                    "analyzer": "standard"
                },
                "`references`": {
                    "type": "text",
                    "analyzer": "standard"
                },
                "appendix": {
                    "type": "text",
                    "analyzer": "standard"
                },
                "semantic_vector": {
                    "type": "dense_vector",
                    "dims": 1024,
                    "index": True,  # 必须设置为 True
                    "similarity": "cosine"  # 指定相似度算法
                }
            }
        }
    }

    # 创建索引
    if not es.indices.exists(index=INDEX_NAME):
        es.indices.create(index=INDEX_NAME, body=mapping)
        print(f"索引 '{INDEX_NAME}' 已创建。")
    else:
        print(f"索引 '{INDEX_NAME}' 已存在。")


def insert_into_es(documents: list):
    # 创建一个 Elasticsearch 客户端实例
    es = Elasticsearch(
        hosts=[{'host': HOST, 'port': PORT, 'scheme': 'http'}],
        # http_auth=('username', 'password'),  # 如果需要认证，请添加此行
        # scheme='https',  # 如果使用 HTTPS，取消此行注释
    )

    # 使用 Bulk API 插入文档到索引中
    actions = [
        {
            "_index": INDEX_NAME,
            "_id": doc["item_id"],  # 可以根据需要使用自定义ID
            "_source": doc
        }
        for doc in documents
    ]

    try:
        # 执行批量插入
        helpers.bulk(es, actions)
        print(f"{len(documents)} 个文档已插入到索引 '{INDEX_NAME}' 中。")
    except helpers.BulkIndexError as e:
        # 捕获批量插入错误，获取详细错误信息
        for error in e.errors:
            print(json.dumps(error, indent=2, ensure_ascii=False))

        # 提取每个错误的具体原因
        for error in e.errors:
            for op_type, details in error.items():
                if not details.get("status") == 200:
                    print(f"Error in document ID {details['_id']}: {details['error']['reason']}")


def load_paper_2_es(paper_info: dict):

    documents = []
    abstract = str(paper_info['abstract'])
    semantic_vector = embed_text(abstract)
    document = {
        "item_id": get_md5(paper_info['title']),
        "title": paper_info['title'],
        "authors": paper_info['authors'],
        "date": paper_info['date'].replace('.', '-'),
        "abstract": paper_info['abstract'],
        "body": paper_info['body'],
        "references": paper_info['references'],
        "appendix": paper_info['appendix'],
        "semantic_vector": semantic_vector
    }
    documents.append(document)
    insert_into_es(documents)


def load_all_papers_2_es(path: str):
    files = list_files_with_os(path)
    index = 0
    for file in files:
        index += 1
        if index >= 355:
            print(f'---------------processing {index} with filename {file}')
            info = extract_pdf_to_json(file)
            output_info = {key: info[key] for key in ['item_id', 'title', 'authors', 'date', 'abstract'] if key in info}
            pretty_result = json.dumps(output_info, indent=4, ensure_ascii=False)
            # 打印格式化后的 JSON
            print(pretty_result)
            load_paper_2_es(info)


if __name__ == '__main__':
    # pdf_path = "./data/papers/多智能体/2023.07.10 RoCo- Dialectic Multi-Robot Collaboration with Large Language
    # Models.pdf"
    #
    # info = extract_pdf_to_json(pdf_path)
    # # create_index()
    # load_paper_2_es(info)
    path = "./data/papers/"
    load_all_papers_2_es(path)
