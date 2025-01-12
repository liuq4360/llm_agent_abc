import threading
import queue
import re
import time
import os
from dotenv_vault import load_dotenv  # pip install --upgrade python-dotenv-vault
load_dotenv()  # https://vault.dotenv.org/ui/ui1
from openai import OpenAI
from configs.model_config import DEEPSEEK_MODEL, DEEPSEEK_BASE_URL
from minimax_tts import call_tts_stream, audio_play

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
)

# 辩论参数
debate_topic = "正方：小孩不能吃糖，反方：小孩能吃糖。"
rules = """
1. 双方轮流发言，正方先开始。
2. 维护自己的观点，反驳对方的观点，禁止人身攻击。
3. 以下情况判定为输：人身攻击、观点前后矛盾、偏离辩题、支持对方观点、5秒内无回应。
4. 辩论结束由裁判判断。
"""
total_rounds = 1  # 设置辩论轮数

# 创建缓冲池
text_buffer = queue.Queue(maxsize=30)  # 设置一个固定大小的队列作为缓冲池
full_response = ""
full_audio = b""


proposition_voice_id = "tianxin_xiaoling"
opposition_voice_id = "male-qn-jingying"
judge_voice_id = "male-qn-jingying"


def fetch_streamed_text(prompt):
    """
    从流式 API 获取文本并放入缓冲池。
    """
    global full_response
    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    for chunk in response:
        if hasattr(chunk, "choices") and chunk.choices[0].delta:  # 确保属性存在
            content = chunk.choices[0].delta.content  # 使用属性访问
            if content:
                full_response += content
                print(content)  # 打印内容
                text_buffer.put(content)


def play_from_buffer(voice_id: str):
    """
    从缓冲池获取文本并进行语音合成和播放。
    """
    global full_audio
    accumulated_text = ""  # 用于累积缓冲池中的文本块
    sentence_endings = "。！？,.，"  # 定义句子结束的标点符号

    while True:
        text = text_buffer.get()  # 从缓冲池取出文本
        if text == "END":  # 特殊标记表示结束
            if accumulated_text:  # 播放剩余的文本
                audio_chunk_iterator = call_tts_stream(accumulated_text, voice_id)
                audio = audio_play(audio_chunk_iterator)
                full_audio += audio
            break

        accumulated_text += text  # 累积文本块

        # 检查累积文本是否包含句子结束标点符号
        if any(accumulated_text.endswith(p) for p in sentence_endings):
            audio_chunk_iterator = call_tts_stream(accumulated_text, voice_id)
            audio = audio_play(audio_chunk_iterator)
            full_audio += audio
            accumulated_text = ""  # 清空已播放的部分


# 辩论过程
class Debate:
    def __init__(self, topic, rules, total_rounds):
        self.topic = topic
        self.rules = rules
        self.history = []
        self.turn = "正方"
        self.judge = "裁判"
        self.ended = False
        self.total_rounds = total_rounds
        self.current_round = 0

    def start(self):
        global full_audio
        start_info = f"""
        辩论主题：{self.topic},
        辩论规则：, {self.rules}.
        现在我们辩论开始，正方先发言。
        """
        audio_chunk_iterator = call_tts_stream(start_info, judge_voice_id)
        audio = audio_play(audio_chunk_iterator)
        full_audio += audio
        print("辩论开始！主题：", self.topic)
        print("规则：", self.rules)
        self.add_to_history("裁判", f"欢迎来到辩论，主题是：{self.topic}。")

        # 开始辩论
        while self.current_round < self.total_rounds and not self.ended:
            self.take_turn()

        # 辩论完成后由裁判作出最终判定
        if not self.ended:
            self.final_judgment()

    def take_turn(self):
        global full_response
        # 当前发言者
        speaker = "正方" if self.turn == "正方" else "反方"
        other = "反方" if self.turn == "正方" else "正方"

        voice_id = proposition_voice_id if self.turn == "正方" else opposition_voice_id

        # 提问生成回复
        prompt = f"你是{speaker}，对方是{other}。当前辩题是：{self.topic}。请发言，维护自己的观点，反驳对方的观点。"

        # 启动线程获取文本并播放音频
        full_response = ""
        fetch_thread = threading.Thread(target=fetch_streamed_text, args=(prompt,))
        play_thread = threading.Thread(target=play_from_buffer, args=(voice_id,))

        # 先启动 fetch_thread
        fetch_thread.start()

        # 启动 play_thread
        play_thread.start()

        # 等待 fetch_thread 完成并在完成时发送结束标记
        fetch_thread.join()
        if not text_buffer.full():
            text_buffer.put("END")  # 添加结束标记

        # 等待 play_thread 完成
        play_thread.join()

        # 更新历史记录
        print(f"--------打印{self.turn}观点------------")
        print(f"{speaker}：\n{full_response}")
        self.add_to_history(speaker, full_response)

        time.sleep(3)

        # 切换发言方
        if self.turn == "反方":
            self.current_round += 1
        self.turn = "反方" if self.turn == "正方" else "正方"

    def final_judgment(self):
        global full_audio

        end_info = f"""
                经过双方激烈的辩论，下面我给出辩论的结果。
                """
        audio_chunk_iterator = call_tts_stream(end_info, judge_voice_id)
        audio = audio_play(audio_chunk_iterator)
        full_audio += audio

        # 提交历史给裁判进行最终判定
        judge_prompt = (
            f"你是裁判。辩题是：{self.topic}。\n以下是历史记录：{self.history}\n请根据辩论表现判断胜负。"
            f"你的输出格式为："
            f"'结论：<正方胜或反方胜>'\n"
            f"'原因：<原因>'")

        # 启动线程获取文本并播放音频
        full_response = ""
        fetch_thread = threading.Thread(target=fetch_streamed_text, args=(judge_prompt,))
        play_thread = threading.Thread(target=play_from_buffer, args=(voice_id,))

        # 先启动 fetch_thread
        fetch_thread.start()

        # 启动 play_thread
        play_thread.start()

        # 等待 fetch_thread 完成并在完成时发送结束标记
        fetch_thread.join()
        if not text_buffer.full():
            text_buffer.put("END")  # 添加结束标记

        # 等待 play_thread 完成
        play_thread.join()

        print(f"---------裁判判定-----------")
        print(f"裁判判定：\n{full_response}")
        self.add_to_history("裁判", full_response)

        # 使用正则表达式提取"结论："后面的结果
        conclusion_match = re.search(r"结论：(正方胜|反方胜)", full_response)
        if conclusion_match:
            conclusion = conclusion_match.group(1)  # 获取结论部分
            if conclusion == "正方胜":
                print("裁判判定：正方胜利")
            elif conclusion == "反方胜":
                print("裁判判定：反方胜利")
        else:
            print("裁判未作出明确判定")
        print("辩论结束！")

    def add_to_history(self, speaker, message):
        self.history.append({"speaker": speaker, "message": message})


# 启动辩论
if __name__ == "__main__":
    debate = Debate(debate_topic, rules, total_rounds)
    debate.start()
    file_name = f'./agent_debate/output/debate.mp3'
    with open(file_name, 'wb') as file:
        file.write(full_audio)
