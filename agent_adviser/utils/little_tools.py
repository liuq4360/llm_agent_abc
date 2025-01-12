import os
import hashlib


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
