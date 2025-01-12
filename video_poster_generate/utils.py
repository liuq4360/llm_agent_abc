import os
import cv2
import json
from PIL import Image


def write_text_to_file(file_path, text, mode="w", encoding="utf-8"):
    """
    将纯文本写入到文本文件中。

    Args:
    - file_path (str): 要写入的文件路径。
    - text (str): 要写入的纯文本内容。
    - mode (str): 文件写入模式，默认为 "w" (覆盖写入)，可以选择 "a" (追加写入)。
    - encoding (str): 文件编码，默认为 "utf-8"。

    Returns:
    - None
    """
    try:
        with open(file_path, mode, encoding=encoding) as file:
            file.write(text)
        print(f"Text successfully written to {file_path}.")
    except Exception as e:
        print(f"An error occurred while writing to the file: {e}")


def save_frames_as_images(frames, output_dir):
    """
    保存相似性最高的帧为图像文件。

    Args:
    - top_frames (list): 包含帧图像数组的列表。
    - output_dir (str): 输出目录路径，用于保存图像。

    Returns:
    - None
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 保存每个帧为图像文件
    for i, frame in enumerate(frames):
        output_path = os.path.join(output_dir, f"poster_{i + 1}.jpg")
        cv2.imwrite(output_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def load_images_as_rgb(candidate_poster_dir):
    """
    从目录中加载所有图片，并将其转换为RGB格式，路径调整为相对当前工作目录的路径。

    Args:
    - candidate_poster_dir (str): 图片所在的目录路径。

    Returns:
    - list: 包含每张图片信息的列表，每个元素是一个字典，格式如下：
      [
        {"image_dir": "图片的相对路径", "frame": 图片的 RGB 格式}
      ]
    """
    # 初始化结果列表
    result = []

    # 获取当前工作目录
    current_dir = os.getcwd()

    # 遍历目录中的所有文件
    for root, _, files in os.walk(candidate_poster_dir):
        for file in files:
            # 构造完整路径
            file_path = os.path.join(root, file)

            try:
                # 打开图片并转换为RGB格式
                with Image.open(file_path) as img:
                    rgb_image = img.convert("RGB")

                # 构造相对于当前工作目录的路径
                relative_path = os.path.relpath(file_path, current_dir)

                # 添加到结果列表
                result.append({"image_dir": relative_path, "frame": rgb_image})

            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

    return result


def save_frames_to_json(frames_with_scores, output_file):
    """
    将包含帧信息及分数的列表保存到 JSON 文件中。

    Args:
    - frames_with_scores (list): 帧信息列表，每个元素为字典，格式如：
      [
          {'image_dir': 'frame1.jpg', 'frame': <PIL.Image.Image>, 'text_similar_score': 0.85, 'quality_score': 0.8},
          {'image_dir': 'frame2.jpg', 'frame': <PIL.Image.Image>, 'text_similar_score': 0.78, 'quality_score': 0.75},
          {'image_dir': 'frame3.jpg', 'frame': <PIL.Image.Image>, 'text_similar_score': 0.90, 'quality_score': 0.9}
      ]
    - output_file (str): 保存的 JSON 文件路径。

    Returns:
    - None
    """
    # 创建一个简化版的列表，只包含可序列化的数据
    serializable_data = [
        {
            'image_dir': frame_info['image_dir'],
            'text_similar_score': frame_info['text_similar_score'],
            'quality_score': frame_info['quality_score']
        }
        for frame_info in frames_with_scores
    ]

    # 保存为 JSON 文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_data, f, indent=4, ensure_ascii=False)

    print(f"Data has been successfully saved to {output_file}")


def read_json_file(file_path):
    """
    从本地 .json 文件中读取数据。

    Args:
    - file_path (str): .json 文件的路径。

    Returns:
    - data (dict or list): 文件中的 JSON 数据。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)  # 解析 JSON 数据
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON. {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    return None
