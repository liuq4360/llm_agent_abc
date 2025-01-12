import base64
from zhipuai import ZhipuAI
import sys
import json
sys.path.append('./')
from utils import read_json_file
import os
from dotenv_vault import load_dotenv  # pip install --upgrade python-dotenv-vault
load_dotenv()  # https://vault.dotenv.org/ui/ui1

ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")


def encode_image(frames: list):
    encode_frames = []
    for img in frames:
        img_path = img["image_dir"]
        with open(img_path, 'rb') as img_file:
            img_base = base64.b64encode(img_file.read()).decode('utf-8')
            img['base64'] = img_base
            encode_frames.append(img)

    return encode_frames


def best_poster_select(encode_frames, text):
    info = []
    for frame in encode_frames:
        info.append(
            {
                "image_url": {
                    "url": frame['base64']
                },
                "type": "image_url"
            }
        )
    num = len(info)
    info.append(
        {
            "text": f"从上面{num}张图片中选择一张海报图，使得这张图最匹配下面一段描述视频概要的文字，"
                    f"你需要综合判断，需要考虑到海报跟视频内容的匹配度、海报的美观度、图片质量、背景合理性、主要对象是否突出等要素。\n"
                    f"{text}\n"
                    f"你的输出是JSON格式，包含2个字段：index、reason。index是给你提供的图片的序号，reason是你选择这张图作为海报的原因。\n"
                    f"你只需要输出图片的序号，给你提供的第1张图片的序号为1，第2张图片的序号为2，以此类推。现在请给出你的输出。",
            "type": "text"
        }
    )

    client = ZhipuAI(api_key=ZHIPU_API_KEY)  # 填写您自己的APIKey
    response = client.chat.completions.create(
        model="glm-4v-plus",  # 填写需要调用的模型名称
        messages=[
            {
                "content": [
                    {
                        "text": "你是一个多媒体专家，擅长基于一段视频的描述文本来选择最匹配该文本的图片，这张图片会作为该视频的海报。",
                        "type": "text"
                    }
                ],
                "role": "assistant"
            },
            {
                "content": info,
                "role": "user"
            }
        ],
        response_format={"type": "json_object"}
    )
    json_string = response.choices[0].message.content
    # 转换为 JSON 对象
    json_object = json.loads(json_string)

    return json_object


if __name__ == "__main__":
    filtered_frames_dir = "./video_poster_generate/output/filtered_frames.json"
    filtered_frames = read_json_file(filtered_frames_dir)
    encode_frames = encode_image(filtered_frames)
    # print(json.dumps(encode_frames, indent=4, ensure_ascii=False))

    summary_text_dir = "./video_poster_generate/output/summary_text.txt"
    # 读取文本
    with open(summary_text_dir, "r", encoding="utf-8") as f:
        text = f.read()

    res = best_poster_select(encode_frames, text)

    index = res['index']
    reason = res['reason']

    print(f"多模态大模型选择的海报图序号为：{index}")
    print(f"多模态大模型选择的海报图的原因为：{reason}")

    candidate_poster = encode_frames[int(index)-1]

    candidate_poster_dir = "./video_poster_generate/output/candidate_poster.json"
    with open(candidate_poster_dir, 'w', encoding='utf-8') as f:
        json.dump(candidate_poster, f, indent=4, ensure_ascii=False)
