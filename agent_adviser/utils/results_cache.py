import redis
import json
import sys
sys.path.append('./')
from agent_adviser.configs.model_config import REDIS_HOST, REDIS_PORT, REDIS_DB_CACHE, CACHE_KEY


# 创建一个 Redis 连接池
pool = redis.ConnectionPool(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB_CACHE, max_connections=100)


def get_redis_connection():
    # 从连接池中获取一个 Redis 连接实例
    return redis.Redis(connection_pool=pool)


def get_json_from_redis():
    # 连接到 Redis 服务器
    r = get_redis_connection()
    # 从 Redis 中获取 JSON 字符串
    json_data = r.get(CACHE_KEY)
    # 将 JSON 字符串转换回 Python 字典
    if json_data is not None:
        return json.loads(json_data)
    return None


def store_json_to_redis(results: dict):
    # 连接到 Redis 服务器
    r = get_redis_connection()
    res = get_json_from_redis()
    # 合并两个字典，后面的值会覆盖前面的
    if res:
        merged_dict = {**res, **results}
    else:
        merged_dict = {**results}
    # 将 Python 字典转换为 JSON 字符串
    json_data = json.dumps(merged_dict)
    # 将 JSON 字符串存储到 Redis 中
    r.set(CACHE_KEY, json_data)


if __name__ == "__main__":
    # res = {
    #     "1+1=?": "1+1=2.",
    #     "agent翻译": "智能体",
    # }
    # store_json_to_redis(res)
    res = get_json_from_redis()
    print(json.dumps(res, indent=4, ensure_ascii=False))
