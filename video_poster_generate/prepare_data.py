import os
import tempfile
from moviepy.video.io.VideoFileClip import VideoFileClip
import whisper
import cv2
import sys
sys.path.append('./')
from openai import OpenAI
from utils import save_frames_as_images, write_text_to_file
import os
from dotenv_vault import load_dotenv  # pip install --upgrade python-dotenv-vault
load_dotenv()  # https://vault.dotenv.org/ui/ui1
from video_poster_generate.configs.model_config import DEEPSEEK_MODEL, DEEPSEEK_BASE_URL

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
)


def generate_summary(content):
    """
    使用 OpenAI 的 GPT 生成文章的中文长文总结（75字）。

    :param content: 文章内容
    :return: 中文长文总结
    """
    prompt = (
        "基于提供的中文内容，请你对内容进行总结。希望你的总结全面、概括，不要遗漏重点信息，请控制在70个字以内。\n"
        f"给你提供的内容是：{content}\n"
        "\n现在请你撰写总结。"
    )
    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": "你是一名专业的文字工作者。"},
            {"role": "user", "content": prompt},
        ],
        max_tokens=4096
    )
    message_content = response.choices[0].message.content.strip()
    return message_content


def extract_audio_as_text(video_path):
    """
    提取视频的音频并使用 Whisper 进行 ASR 转换。
    :param video_path:
    :return:
    """
    # 提取音频
    video_clip = VideoFileClip(video_path)
    temp_audio = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    video_clip.audio.write_audiofile(temp_audio.name)

    # 使用 Whisper 进行 ASR 转换
    model = whisper.load_model("large")
    result = model.transcribe(temp_audio.name)
    os.remove(temp_audio.name)
    return result["text"]


def extract_high_quality_frames(video_path, frame_interval=30, quality_threshold=100.0):
    """
        从视频中提取质量较高的帧作为候选帧。
        视频通过帧间隔采样，跳过不需要处理的帧。
        使用 Laplacian 方差法 计算帧的清晰度。
        清晰度超过指定阈值的帧会被提取并存储。
    :param video_path:
    :param frame_interval:
    :param quality_threshold:
    :return:
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_quality = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 仅处理间隔帧
        if frame_count % frame_interval == 0:
            # 计算清晰度（Laplacian 方差法）
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            quality = cv2.Laplacian(gray, cv2.CV_64F).var()
            if quality >= quality_threshold:
                # 转换颜色通道顺序为 RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                frame_quality.append(quality)

        frame_count += 1

    cap.release()
    return frames, frame_quality


if __name__ == "__main__":
    video_path = "./video_poster_generate/data/1.mp4"  # 替换为你的视频路径
    candidate_poster_dir = "./video_poster_generate/output/candidate_frames"
    os.makedirs(candidate_poster_dir, exist_ok=True)
    summary_text_dir = "./video_poster_generate/output/summary_text.txt"

    text = extract_audio_as_text(video_path)
    summary_text = generate_summary(text)  # 对文本进行摘要，控制字数（clip模型要求控制字数）同时突出重点。
    print(f"summary_text: {summary_text}")

    write_text_to_file(summary_text_dir, summary_text)

    frames, _ = extract_high_quality_frames(video_path, frame_interval=30, quality_threshold=100.0)

    save_frames_as_images(frames, candidate_poster_dir)
