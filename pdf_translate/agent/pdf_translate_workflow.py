import sys
import json
sys.path.append('./')
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from pdf_translate.utils.extract_text_from_pdf import extract_text_by_paragraph
from pdf_translate.utils.extract_images_from_pdf import extract_images
from pdf_translate.utils.translate_pages_and_paragraphs import translate_all_pages
from pdf_translate.utils.draw_text_on_pdf import generate_pdf_with_text
from pdf_translate.utils.draw_image_on_pdf import draw_images_on_pdf


def read_json_from_file(file_path):
    """
    从文件中读取 JSON 数据，并转换为 Python 对象。
    :param file_path: 文件路径
    :return: JSON 对象（字典或列表）
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)  # 将文件内容解析为 JSON 对象
        return data
    except json.JSONDecodeError as e:
        print(f"JSON 格式错误: {e}")
        return None
    except FileNotFoundError:
        print("文件未找到，请检查文件路径")
        return None
    except Exception as e:
        print(f"发生未知错误: {e}")
        return None


# 定义工作流的状态类，继承 Pydantic 的 BaseModel
class WorkflowState(BaseModel):
    pdf_file: str = Field(default_factory=str, description="需要翻译的英文PDF")
    text_json_file: str = Field(default_factory=str, description="抽取的文本对应的JSON文件")
    image_path: str = Field(default_factory=str, description="抽取的图像保存的目录")
    image_json_file: str = Field(default_factory=str, description="抽取的图片对应的JSON文件")
    translated_text_json_file: str = Field(default_factory=str, description="抽取的文本翻译后对应的JSON文件")
    translated_pdf_file: str = Field(default_factory=str, description="翻译后的PDF")
    translated_pdf_with_image_file: str = Field(default_factory=str, description="翻译后并且补充了图片的PDF")


# 节点1：从pdf中抽取文字
def extract_text(state):
    print("从英文PDF中提取对应的文字...")
    text_data = extract_text_by_paragraph(state.pdf_file)
    with open(state.text_json_file, "w", encoding="utf-8") as f:
        json.dump(text_data, f, indent=4, ensure_ascii=False)


# 节点2：从pdf中抽取图片
def extract_image(state):
    print("从英文PDF中提取图片 ...")
    image_metadata = extract_images(state.pdf_file, state.image_path)
    with open(state.image_json_file, "w", encoding="utf-8") as f:
        json.dump(image_metadata, f, indent=4, ensure_ascii=False)


# 节点3：文字翻译为中文
def translate_text(state):

    print("翻译英文为中文 ...")
    text_with_layout = read_json_from_file(state.text_json_file)
    translate_all_pages(text_with_layout, state.translated_text_json_file)


# 节点4：生成带翻译后文字的pdf
def draw_text_pdf(state):
    print("基于翻译后的中文，构建新PDF ...")
    translated_text_with_layout = read_json_from_file(state.translated_text_json_file)
    generate_pdf_with_text(translated_text_with_layout, state.translated_pdf_file)


# 节点5：从生成的带文字的pdf中补充对应的图片
def draw_images_pdf(state):
    print("翻译后的PDF补充图片 ...")
    images_data = read_json_from_file(state.image_json_file)
    draw_images_on_pdf(state.translated_pdf_file, state.translated_pdf_with_image_file, images_data)


# 定义工作流图
def build_workflow():
    workflow = StateGraph(WorkflowState)

    workflow.add_node("extract_text", extract_text)
    workflow.add_node("extract_image", extract_image)
    workflow.add_node("translate_text", translate_text)
    workflow.add_node("draw_text_pdf", draw_text_pdf)
    workflow.add_node("draw_images_pdf", draw_images_pdf)

    # extract_text 和 extract_image 没有依赖，可以同时启动。
    workflow.add_edge(START, "extract_text")
    workflow.add_edge(START, "extract_image")

    workflow.add_edge("extract_text", "translate_text")  # 需要先抽取文本，然后才能翻译文本
    workflow.add_edge("translate_text", "draw_text_pdf")  # 需要先翻译文本，才能绘制文本PDF
    workflow.add_edge("draw_text_pdf", "draw_images_pdf")  # 需要先绘制文本PDF，才能绘制包含图片的PDF
    workflow.add_edge("draw_images_pdf", END)

    return workflow.compile()


# 运行工作流
if __name__ == "__main__":
    # 初始化状态
    initial_state = WorkflowState()
    initial_state.pdf_file = "./pdf_translate/data/o1-system-card-20241205.pdf"
    initial_state.text_json_file = "./pdf_translate/output/o1-system-card-20241205.json"
    initial_state.image_path = "./pdf_translate/output/images/"
    initial_state.image_json_file = "./pdf_translate/output/o1-system-card-20241205-images-info.json"
    initial_state.translated_text_json_file = "./pdf_translate/output/o1-system-card-20241205-translated.json"
    initial_state.translated_pdf_file = "./pdf_translate/output/o1-system-card-中文.pdf"
    initial_state.translated_pdf_with_image_file = "./pdf_translate/output/o1-system-card-中文-带图.pdf"

    # 编译并运行工作流
    workflow_app = build_workflow()
    print("🚀 开始执行工作流...")
    final_state = workflow_app.invoke(initial_state)
    print("✅ 工作流执行完毕！")
