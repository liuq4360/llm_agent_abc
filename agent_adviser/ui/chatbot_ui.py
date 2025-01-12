import gradio as gr
import sys
sys.path.append('./')
from agent_adviser.configs.model_config import SEARCH_RESULTS_NUM, SIMILARITY_THRESHOLD
from agent_adviser.serving.search_pipeline import search_api
from agent_adviser.utils.prompt_build import build_llm_prompt
from agent_adviser.serving.nlu.semantics_classify import semantics_classify
from agent_adviser.serving.chat.streaming_chat import deepseek_chat
from agent_adviser.utils.results_cache import get_json_from_redis, store_json_to_redis
from agent_adviser.utils.little_tools import find_most_similar


def chat(user_input, info_display, history):
    info_display.append((user_input, None))  # 用户输入立即展示
    history.append((user_input, None))  # 历史记录中暂时保留用户输入
    tmp_btn = gr.Button("👍 赞同", elem_id="like_button", visible=False)
    yield info_display, info_display, history, "", tmp_btn  # 清空输入框，立即更新界面

    prompt = user_input
    cache_dict = get_json_from_redis()

    index, text, score = find_most_similar(str(user_input), list(cache_dict.keys()))
    if score > SIMILARITY_THRESHOLD:  # 当相似等分大于此值时，认为这个问题已经被回答过了，可以直接从缓存中返回结果了。
        response_content = cache_dict[text]
        # **第四步：记录完整的AI回答**
        info_display[-1] = (user_input, response_content)  # 确保完整记录
        history[-1] = (prompt, response_content)  # 更新历史记录
        tmp_btn = gr.Button("👍 赞同", elem_id="like_button", visible=False)
        yield info_display, info_display, history, "", tmp_btn  # 清空输入框

    else:
        class_ = semantics_classify(user_input)
        if class_ == "专业咨询":  # 如果是专业咨询，调用search_api
            rerank_results = search_api(user_input, SEARCH_RESULTS_NUM)
            prompt = build_llm_prompt(user_input, rerank_results)

        history[-1] = (prompt, None)  # 更新prompt到最新内容，利用召回的信息做为背景材料，将用户的问题转为prompt。
        response_content = ""
        for partial_response in deepseek_chat(history):
            # 更新prompt并生成AI回复
            response_content = partial_response  # 流式获取AI回复
            info_display[-1] = (user_input, response_content)  # 更新AI部分回复
            tmp_btn = gr.Button("👍 赞同", elem_id="like_button", visible=False)
            yield info_display, info_display, history, "", tmp_btn  # 实时更新界面

        info_display[-1] = (user_input, response_content)  # 确保完整记录
        history[-1] = (prompt, response_content)  # 更新历史记录
        tmp_btn = gr.Button("👍 赞同", elem_id="like_button", visible=True)
        yield info_display, info_display, history, "", tmp_btn  # 清空输入框


def cache_agent_answer(info_display):
    # 如果对话记录中有内容，则显示按钮，用户可以点赞，将这条回答缓存到Redis中，后续有类似问题，可以直接从缓存中获取答案。
    if info_display:
        question, answer = info_display[-1]

        store_json_to_redis({
            question: answer
        })

    tmp_btn = gr.Button("👍 赞同", elem_id="like_button", visible=False)
    return tmp_btn


# 创建Gradio的UI组件
with gr.Blocks() as demo:
    with gr.Row():  # 顶部容器
        # 使用Flexbox布局让页面占满
        with gr.Column(scale=12):  # 用一个column占满剩余空间
            # 新建一个容器包裹chatbot，方便控制高度
            with gr.Row(elem_id="chatbot-container"):
                chatbot = gr.Chatbot(elem_id="chatbot")  # 创建聊天框
            with gr.Row(elem_id="like-container"):
                with gr.Row():
                    with gr.Column(scale=1):
                        like_btn = gr.Button("👍 赞同", elem_id="like_button", visible=False)
            with gr.Row(elem_id="bottom-row"):  # 底部容器
                with gr.Row():
                    with gr.Column(scale=0.85):
                        txt = gr.Textbox(show_label=False, placeholder="请输入你的问题...", elem_id="textbox")
                    with gr.Column(scale=0.15):
                        btn = gr.Button("提交", elem_id="button")

    # 状态变量
    info_display = gr.State([])  # 聊天记录
    history = gr.State([])  # 历史记录

    # 点击事件逻辑
    btn.click(chat, inputs=[txt, info_display, history], outputs=[chatbot, info_display, history,
                                                                  txt, like_btn], queue=True)
    txt.submit(chat, inputs=[txt, info_display, history], outputs=[chatbot, info_display, history,
                                                                   txt, like_btn], queue=True)
    like_btn.click(cache_agent_answer, inputs=[info_display], outputs=[like_btn], queue=True)

# 启动应用
demo.queue().launch()
