import sys
import torch
import numpy as np
sys.path.append('./')
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer
from utils import load_images_as_rgb, save_frames_to_json
import torchvision.transforms as transforms
from video_poster_generate.configs.model_config import BEST_POSTER_NUM


def get_text_similarity_scores(frames_with_paths, text, batch_size=50):
    """
    使用 CLIP 模型对帧和文本进行匹配，并用 Softmax 对分值进行归一化。

    Args:
    - frames_with_paths (list): 每个元素为字典，格式：
      [
          {'image_dir': "图片的相对路径", "frame": 图片的 RGB 格式},
          ...
      ]
    - text (str): 要匹配的文本。
    - batch_size (int): 每批处理的帧数。

    Returns:
    - result (list): 包含每帧路径、RGB 图像和归一化文本相似度分数的列表，格式：
      [
          {'image_dir': "图片的相对路径", "frame": 图片的 RGB 格式, "text_similar_score": 归一化相似度分数},
          ...
      ]
    """
    # 加载 CLIP 模型和处理器
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    # 获取模型支持的最大文本长度
    max_text_length = model.config.text_config.max_position_embeddings

    # 截断文本以适应最大长度
    if len(text) > max_text_length:
        text = text[:max_text_length]

    # 准备数据
    pil_frames = [frame_info["frame"] for frame_info in frames_with_paths]
    image_dirs = [frame_info["image_dir"] for frame_info in frames_with_paths]

    # 分批处理帧
    logits_per_image = []
    for i in range(0, len(pil_frames), batch_size):
        batch_frames = pil_frames[i:i + batch_size]
        inputs = processor(text=[text], images=batch_frames, return_tensors="pt", padding=True, truncation=True)

        # 模型推理
        with torch.no_grad():
            outputs = model(**inputs)
        logits_per_image_batch = outputs.logits_per_image.detach().numpy()
        logits_per_image.extend(logits_per_image_batch.squeeze().tolist())  # 累积结果

    # Min-Max 归一化分值
    logits_per_image = np.array(logits_per_image)
    min_score = np.min(logits_per_image)
    max_score = np.max(logits_per_image)

    # 防止分母为0的情况
    if max_score - min_score == 0:
        normalized_scores = np.ones_like(logits_per_image)  # 如果所有值相同，归一化为1
    else:
        normalized_scores = (logits_per_image - min_score) / (max_score - min_score)

    # 生成结果
    result = [
        {
            "image_dir": image_dirs[i],
            "frame": pil_frames[i],
            "text_similar_score": round(normalized_scores[i], 4)
        }
        for i in range(len(pil_frames))
    ]

    return result


def get_quality_scores(frames_with_info):
    """
    为一组帧计算质量分数，并返回包含质量分数的帧信息。

    Args:
    - frames_with_info (list): 每个元素为字典，格式：
      [
          {'image_dir': 'frame1.jpg', 'frame': <PIL.Image.Image>, 'text_similar_score': 0.85},
          {'image_dir': 'frame2.jpg', 'frame': <PIL.Image.Image>, 'text_similar_score': 0.78},
          {'image_dir': 'frame3.jpg', 'frame': <PIL.Image.Image>, 'text_similar_score': 0.90}
      ]

    Returns:
    - results (list): 包含质量分数的帧信息列表，格式：
      [
          {'image_dir': 'frame1.jpg', 'frame': <PIL.Image.Image>, 'text_similar_score': 0.85, 'quality_score': 0.8},
          {'image_dir': 'frame2.jpg', 'frame': <PIL.Image.Image>, 'text_similar_score': 0.78, 'quality_score': 0.75},
          {'image_dir': 'frame3.jpg', 'frame': <PIL.Image.Image>, 'text_similar_score': 0.90, 'quality_score': 0.9}
      ]
    """
    # 检查是否有可用的 GPU
    device = torch.device("cuda") if torch.cuda.is_available() else "mps"

    # 加载模型
    model = torch.hub.load(
        repo_or_dir="miccunifi/ARNIQA",
        source="github",
        model="ARNIQA",
        regressor_dataset="kadid10k",  # 可选其他数据集
    )
    model.eval().to(device)

    # 定义预处理流水线
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 初始化结果存储
    results = []

    # 遍历所有帧并计算分数
    for frame_info in frames_with_info:
        image_dir = frame_info['image_dir']
        frame = frame_info['frame']
        text_similar_score = frame_info['text_similar_score']

        # 获取帧的半分辨率版本
        img_ds = transforms.Resize((frame.size[1] // 2, frame.size[0] // 2))(frame)

        # 预处理
        img = preprocess(frame).unsqueeze(0).to(device)
        img_ds = preprocess(img_ds).unsqueeze(0).to(device)

        # 计算质量得分
        with torch.no_grad(), torch.cuda.amp.autocast():
            quality_score = model(img, img_ds, return_embedding=False, scale_score=True).item()

        # 添加结果
        results.append({
            'image_dir': image_dir,
            'frame': frame,
            'text_similar_score': text_similar_score,
            'quality_score': round(quality_score, 4)
        })

    return results


def filter_and_sort_frames(frames, best_poster_num=5):
    """
    根据指定条件筛选并排序满足条件的帧。

    Args:
    - frames (list): 输入的帧列表，每个帧为包含以下键的字典：
      [
          {'image_dir': 'frame1.jpg', 'frame': <PIL.Image.Image>, 'text_similar_score': 0.85, 'quality_score': 0.8},
          {'image_dir': 'frame2.jpg', 'frame': <PIL.Image.Image>, 'text_similar_score': 0.78, 'quality_score': 0.75},
          {'image_dir': 'frame3.jpg', 'frame': <PIL.Image.Image>, 'text_similar_score': 0.90, 'quality_score': 0.9}
      ]
    - min_similarity_score (float): 最小文本相似度分数。
    - min_quality_score (float): 最小质量分数。
    - best_poster_num (int): 返回的最大帧数量。

    Returns:
    - filtered_frames (list): 筛选后的帧列表，每个元素为原始字典格式。
    """
    # 1. 按照调和平均数排序
    sorted_frames = sorted(
        frames,
        key=lambda x: (
                2 * x['text_similar_score'] * x['quality_score'] /
                (x['text_similar_score'] + x['quality_score'] if x['text_similar_score'] + x[
                    'quality_score'] > 0 else 1)
        ),
        reverse=True
    )

    # 2. 限制返回帧的数量不超过 `best_poster_num`
    return sorted_frames[:best_poster_num]


if __name__ == "__main__":
    candidate_poster_dir = "./video_poster_generate/output/candidate_frames"
    summary_text_dir = "./video_poster_generate/output/summary_text.txt"
    # 读取文本
    with open(summary_text_dir, "r", encoding="utf-8") as f:
        text = f.read()

    frames_with_paths = load_images_as_rgb(candidate_poster_dir)

    frames_with_info = get_text_similarity_scores(frames_with_paths, text)

    frames_with_scores = get_quality_scores(frames_with_info)

    frames_dir = "./video_poster_generate/output/frames_with_scores.json"
    save_frames_to_json(frames_with_scores, frames_dir)

    filtered_frames = filter_and_sort_frames(frames_with_scores, BEST_POSTER_NUM)

    filtered_frames_dir = "./video_poster_generate/output/filtered_frames.json"
    save_frames_to_json(filtered_frames, filtered_frames_dir)
