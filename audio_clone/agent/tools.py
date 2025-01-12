from langchain_core.tools import BaseTool
import noisereduce as nr
import librosa
import soundfile as sf
import whisper
import requests
from openai import OpenAI
import re
from pydub import AudioSegment
import os
from dotenv_vault import load_dotenv  # pip install --upgrade python-dotenv-vault
load_dotenv()  # https://vault.dotenv.org/ui/ui1
from audio_clone.configs.model_config import STEP_BASE_URL

STEP_KEY = os.getenv("STEP_KEY")


def split_text_into_segments(text, max_length=1000):
    """
    将一段中文文本按照完整句子分割成多段，每段长度不超过指定字符数。

    :param text: 输入的中文文本
    :param max_length: 每段最大字符数，默认为 1000
    :return: 分割后的文本段列表
    """
    if not text:
        return []

    # 使用正则表达式按照句子结束符进行分割
    sentences = re.split(r'(。|！|？|\.|\!|\?)', text)

    # 合并分割结果，确保句子完整
    segments = []
    current_segment = ""
    for i in range(0, len(sentences) - 1, 2):
        sentence = sentences[i] + (sentences[i + 1] if i + 1 < len(sentences) else "")
        if len(current_segment) + len(sentence) <= max_length:
            current_segment += sentence
        else:
            segments.append(current_segment)
            current_segment = sentence

    # 添加最后一段
    if current_segment:
        segments.append(current_segment)

    return segments


def merge_audio_files(files, output_file):
    """
    合并多个音频文件并保存为一个文件。

    :param files: 音频文件路径的列表，例如 ["file1.mp3", "file2.mp3", "file3.mp3"]
    :param output_file: 输出合并后的音频文件路径
    """
    if not files or len(files) < 2:
        raise ValueError("请提供至少两个音频文件进行合并")

    # 加载第一个音频文件作为基础
    combined_audio = AudioSegment.from_file(files[0], format="mp3")

    # 依次加载并合并其他音频文件
    for file in files[1:]:
        audio = AudioSegment.from_file(file, format="mp3")
        combined_audio += audio  # 顺序合并

    # 保存合并后的音频文件
    combined_audio.export(output_file, format="mp3")
    print(f"音频已成功合并并保存为: {output_file}")


class ExtractAudioTool(BaseTool):
    name: str = "extract_audio"
    description: str = ("Extracts the first 10 seconds of an audio file, Input should be a JSON with keys "
                        "audio_path, output_audio_file and output_text_file. The return of this tools should be a "
                        "JSON with keys output_audio_file and output_text_file, the output_audio_file is the "
                        "extracted audio file path, the output_text_file is the extracted audio's text file path.")

    def _run(self, inputs: dict) -> dict:
        audio_path = inputs['audio_path']
        output_audio_file = inputs['output_audio_file']
        output_text_file = inputs['output_text_file']
        whisper_model = "large"
        # 1. 提取音频的前10秒
        print("Extracting first 10 seconds of audio...")
        audio = AudioSegment.from_file(audio_path)
        first_10_seconds = audio[:10000]  # 单位为毫秒
        temp_audio_file = "temp_10s.wav"
        first_10_seconds.export(temp_audio_file, format="wav")

        # 2. 去掉背景声音
        print("Reducing background noise...")
        y, sr_rate = librosa.load(temp_audio_file, sr=None)  # 加载音频
        noise_profile = y[:sr_rate]
        reduced_noise_audio = nr.reduce_noise(y=y, sr=sr_rate, y_noise=noise_profile)
        sf.write(output_audio_file, reduced_noise_audio, sr_rate)

        # 删除临时文件
        os.remove(temp_audio_file)

        # 3. 转录音频为文字
        print("Transcribing audio with Whisper...")
        model = whisper.load_model(whisper_model)
        result = model.transcribe(output_audio_file, language="zh")
        voice_text = result["text"]

        # 保存文字到文件
        with open(output_text_file, "w", encoding="utf-8") as f:
            f.write(voice_text)

        print("Text successfully extracted and saved.")

        outputs = {
            "output_audio_file": output_audio_file,
            "output_text_file": output_text_file
        }

        return outputs


class UploadAudioTool(BaseTool):
    name: str = "upload_audio"
    description: str = ("Uploads audio(output_audio_file) to cloud storage and returns "
                        "the file id：voice_file_id.")

    def _run(self, output_audio_file: str) -> str:
        client = OpenAI(api_key=STEP_KEY, base_url="https://api.stepfun.com/v1")

        response = client.files.create(
            file=open(output_audio_file, "rb"),
            purpose="storage"
        )

        # 获取文件 ID 和文件名
        voice_file_id = response.id
        # filename = response.filename
        # print(f"File ID: {file_id}, Filename: {filename}")
        return voice_file_id


class CloneVoiceTool(BaseTool):
    name: str = "clone_voice"
    description: str = (
        "Clones the voice from an uploaded audio file id and voice text,"
        "Input should be a JSON with keys voice_file_id and output_text_file."
        "return the cloned voice id：cloned_voice_id.")

    def _run(self, inputs: dict) -> str:
        voice_file_id = inputs['voice_file_id']
        output_text_file = inputs['output_text_file']

        # 读取文本文件内容
        with open(output_text_file, "r", encoding="utf-8") as file:
            voice_text = file.read()

        url = "https://api.stepfun.com/v1/audio/voices"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {STEP_KEY}"  # 替换为实际的 API Key
        }
        data = {
            "file_id": voice_file_id,
            "model": "step-tts-mini",
            "text": voice_text
        }

        response = requests.post(url, headers=headers, json=data)

        # 打印响应结果
        if response.status_code == 200:
            print("成功：",
                  response.json())  # {'id': 'voice-tone-DCV8jkPKFM', 'object': 'audio.voice', 'duplicated': True}
            cloned_voice_id = response.json()['id']
            return cloned_voice_id
        else:
            print("失败：", response.status_code, response.text)
            return ""


class GenerateVoiceTool(BaseTool):
    name: str = "generate_voice"
    description: str = ("Generates audio from clone_text using the cloned_voice_id, the generated audio store in a "
                        "local file（tts_file_path）. Input should be a JSON with keys cloned_voice_id, clone_text "
                        "and tts_file_path.")

    def _run(self, inputs: dict) -> None:
        cloned_voice_id = inputs['cloned_voice_id']
        clone_text = inputs['clone_text']
        tts_file_path = inputs['tts_file_path']
        client = OpenAI(
            api_key=STEP_KEY,
            base_url=STEP_BASE_URL
        )
        segments = split_text_into_segments(text=clone_text, max_length=1000)
        if len(segments) == 1:
            response = client.audio.speech.create(
                model="step-tts-mini",
                voice=cloned_voice_id,
                input=clone_text
            )
            response.stream_to_file(tts_file_path)
        else:
            audio_files = []
            for idx, seg in enumerate(segments):
                response = client.audio.speech.create(
                    model="step-tts-mini",
                    voice=cloned_voice_id,
                    input=seg
                )
                temp_audio_file = f"{idx}.wav"
                response.stream_to_file(temp_audio_file)
                audio_files.append(temp_audio_file)

            merge_audio_files(audio_files, tts_file_path)
            for temp_audio_file in audio_files:
                os.remove(temp_audio_file)
