# This Python file uses the following encoding: utf-8

import json
import subprocess
import time
from typing import Iterator
import requests
import os
from dotenv_vault import load_dotenv  # pip install --upgrade python-dotenv-vault
load_dotenv()  # https://vault.dotenv.org/ui/ui1
from configs.model_config import MINIMAX_URL_PREFIX

MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY")
MINIMAX_GROUP_ID = os.getenv("MINIMAX_GROUP_ID")
MINIMAX_URL = MINIMAX_URL_PREFIX + MINIMAX_GROUP_ID


file_format = 'mp3'  # 音频格式，支持 mp3/pcm/flac


def build_tts_stream_headers() -> dict:
    headers = {
        'accept': 'application/json, text/plain, */*',
        'content-type': 'application/json',
        'authorization': "Bearer " + MINIMAX_API_KEY,
    }
    return headers


def build_tts_stream_body(text: str, voice_id: str) -> str:
    body = json.dumps({
        "model": "speech-01-turbo",  # 模型名称
        "text": text,   # 需要生成语音的文本
        "stream": True,  # 是否是流式输出，我们选择是。
        "voice_setting": {
            "voice_id": voice_id,  # 声音id，可以在minimax官网查看，即参考文献1
            "speed": 1,
            "vol": 1,
            "pitch": 0,
            "emotion": "neutral"  # 声音的情感，这里选择中性的。
        },
        "pronunciation_dict": {
            "tone": [
                "处理/(chu3)(li3)", "危险/dangerous"
            ]
        },
        "audio_setting": {
            "audio_sample_rate": 32000,
            "bitrate": 128000,
            "format": file_format,
            "channel": 1
        }
    })
    return body


mpv_command = ["mpv", "--no-cache", "--no-terminal", "--", "fd://0"]  # mpv是一个开源的音频播放器，我们使用它来播放音频。需要你自己安装。
mpv_process = subprocess.Popen(
    mpv_command,
    stdin=subprocess.PIPE,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)


def call_tts_stream(text: str, voice_id: str) -> Iterator[bytes]:
    tts_headers = build_tts_stream_headers()
    tts_body = build_tts_stream_body(text, voice_id)

    response = requests.request("POST", MINIMAX_URL, stream=True, headers=tts_headers, data=tts_body)
    for chunk in response.raw:
        if chunk:
            if chunk[:5] == b'data:':
                data = json.loads(chunk[5:])
                if "data" in data and "extra_info" not in data:
                    if "audio" in data["data"]:
                        audio = data["data"]['audio']
                        yield audio


def audio_play(audio_stream: Iterator[bytes]) -> bytes:
    audio = b""
    for chunk in audio_stream:
        if chunk is not None and chunk != '\n':
            decoded_hex = bytes.fromhex(chunk)
            mpv_process.stdin.write(decoded_hex)  # type: ignore
            mpv_process.stdin.flush()
            audio += decoded_hex

    return audio


if __name__ == "__main__":
    text = "真正的危险不是计算机开始像人一样思考，而是人开始像计算机一样思考。计算机只是可以帮我们处理一些简单事务。"
    voice_id = "female-tianmei"
    audio_chunk_iterator = call_tts_stream(text, voice_id)
    audio = audio_play(audio_chunk_iterator)

    # 结果保存至文件
    timestamp = int(time.time())
    file_name = f'output_total_{timestamp}.{file_format}'
    with open(file_name, 'wb') as file:
        file.write(audio)
