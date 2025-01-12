import json
import time
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from tools import ExtractAudioTool, UploadAudioTool, CloneVoiceTool, GenerateVoiceTool
from langgraph.checkpoint.memory import MemorySaver
import os
from dotenv_vault import load_dotenv  # pip install --upgrade python-dotenv-vault
load_dotenv()  # https://vault.dotenv.org/ui/ui1
from audio_clone.configs.model_config import STEP_BASE_URL

memory = MemorySaver()


STEP_KEY = os.getenv("STEP_KEY")


# 定义工具列表
tools = [
    ExtractAudioTool(),
    UploadAudioTool(),
    CloneVoiceTool(),
    GenerateVoiceTool()
]


model = ChatOpenAI(model="step-1-32k", api_key=STEP_KEY, base_url=STEP_BASE_URL)
# step-1-128k
# step-1-256k

agent_executor = create_react_agent(model, tools, checkpointer=memory)

config = {"configurable": {"thread_id": "abc123"}}

TEXT = """
高斯为中心的端到端自动驾驶：如何突破传统方法的局限？

自动驾驶技术近年来取得了显著进展，尤其是在基于视觉的自动驾驶领域。然而，现有的方法在处理复杂场景时，往往面临效率与全面性之间的权衡。传统的密集表示（如鸟瞰图）或稀疏表示（如实例框）虽然各有优势，但都无法同时兼顾高效性和细节捕捉。为了解决这一问题，来自清华大学、理想汽车和北京大学的研究团队提出了一种全新的端到端自动驾驶框架——GaussianAD，该框架以高斯分布为核心，通过稀疏但全面的3D高斯表示来描述场景，从而在效率和细节捕捉之间找到了平衡。

1. 背景与挑战

传统方法的局限

传统的自动驾驶系统通常采用模块化设计，分为感知、预测和规划三个阶段。感知阶段通过摄像头或激光雷达获取环境信息，预测阶段根据感知结果推测未来的场景变化，规划阶段则根据预测结果生成车辆的行驶路径。然而，这种模块化设计存在信息传递中的损失问题，尤其是在从2D图像到3D场景的转换过程中，信息压缩可能导致关键细节的丢失。

近年来，端到端自动驾驶方法逐渐兴起，旨在直接从图像输入中生成未来的行驶轨迹。然而，现有的端到端方法在中间场景表示上仍然存在不足。例如，鸟瞰图（BEV）表示虽然高效，但无法捕捉到场景的精细3D结构；而稀疏查询（如实例框）虽然节省计算资源，但难以描述复杂的3D环境。此外，一些方法采用体素（voxel）或三视角视图（tri-perspective view）来表示3D占用空间，虽然能捕捉更多细节，但计算开销巨大，难以在实际应用中推广。

高斯分布的优势

为了解决上述问题，研究团队提出了一种基于3D语义高斯分布的场景表示方法。高斯分布具有稀疏性和通用逼近性的特点，能够以较少的计算资源捕捉到场景的精细结构。通过将场景中的物体和环境元素表示为3D高斯分布，系统可以在保持高效性的同时，提供更全面的场景信息。

2. GaussianAD的核心创新

3D高斯场景表示

GaussianAD的核心创新在于其3D高斯场景表示。系统首先将场景初始化为均匀分布的3D高斯点，然后通过周围视图的图像逐步对其进行细化，最终得到一个稀疏但全面的3D高斯场景表示。这种表示方法不仅能够捕捉到场景中的动态和静态元素，还能为后续的感知、预测和规划任务提供丰富的信息。

稀疏卷积与3D感知

在感知阶段，GaussianAD将3D高斯点视为语义点云，并采用稀疏卷积技术进行高效处理。稀疏卷积是一种专门针对稀疏数据设计的卷积方法，能够在保持计算效率的同时，处理大规模的3D数据。通过这种方式，系统能够高效地完成3D检测和语义地图构建等任务。

3D高斯流与未来预测

为了预测未来的场景变化，GaussianAD引入了3D高斯流的概念。系统为每个高斯点预测一个未来的位移，从而模拟场景中动态元素（如车辆、行人）的运动轨迹。通过整合所有高斯点的预测结果，系统能够生成未来的场景预测，并据此规划车辆的行驶路径。

端到端训练与监督

GaussianAD的另一个亮点在于其端到端的训练方式。系统可以直接从图像输入中学习到场景表示、未来预测和路径规划的全过程。此外，当感知标签可用时，系统还可以选择性地引入监督信号，进一步提升模型的性能。

3. 实验与结果

数据集与评估

研究团队在广泛使用的nuScenes数据集上对GaussianAD进行了广泛的实验验证。nuScenes数据集包含了丰富的自动驾驶场景，涵盖了多种天气、光照和交通条件，是评估自动驾驶系统性能的理想平台。

任务表现

实验结果表明，GaussianAD在多个任务上均表现出色。在运动规划任务中，GaussianAD的规划路径更加平滑且符合实际驾驶需求；在3D占用预测任务中，系统能够准确预测场景中的动态和静态元素；在4D占用预测任务中，GaussianAD能够提前预测未来的场景变化，为车辆的安全行驶提供保障。

效率与精度

与传统方法相比，GaussianAD在保持高精度的同时，显著降低了计算开销。通过稀疏的高斯表示和稀疏卷积技术，系统能够在有限的计算资源下完成复杂的场景分析和路径规划任务。

4. 结论与未来展望

突破传统方法的局限

GaussianAD通过引入3D高斯分布作为场景表示的核心，成功突破了传统方法在效率与全面性之间的权衡。稀疏的高斯表示不仅能够高效地捕捉场景的精细结构，还能为后续的感知、预测和规划任务提供丰富的信息。

未来的研究方向

尽管GaussianAD在当前实验中表现出色，但研究团队认为仍有进一步优化的空间。例如，未来的研究可以探索如何进一步减少高斯点的数量，同时保持场景表示的精度；此外，如何将GaussianAD应用于更复杂的实际驾驶场景，也是一个值得深入研究的方向。

5. 总结

GaussianAD的提出为自动驾驶技术的发展开辟了新的道路。通过以高斯分布为核心的端到端框架，系统能够在保持高效性的同时，提供更全面的场景信息。实验结果表明，GaussianAD在多个任务上均达到了行业领先水平，为未来的自动驾驶技术发展提供了新的思路。

无论是从技术角度还是从实际应用的角度，GaussianAD都展示了巨大的潜力。随着技术的不断进步，我们有理由相信，基于高斯分布的自动驾驶系统将在未来的智能交通中发挥越来越重要的作用。
"""


content = f"""你是一个按照要求完成任务的智能体。 

你的工作流程如下：

首先，你需要根据给定的音频文件（audio_path），首先截取该文件的前10s，生成10s音频文件（output_audio_file）和对应的10s音频文字（output_text_file）。然后将10s音频上传到云端，获得对应的文件id（voice_file_id）。

当你上传完成后，你需要将10s音频文件对应的文件id（voice_file_id）和10s音频文件对应的文字（output_text_file）作为输入，利用这个文件id和文字来克隆对应的声音（cloned_voice_id）。

当你克隆完成后，你需要将下面这段话利用克隆的cloned_voice_id来生成对应的语音文件。

给你提供的待生成tts文件的文本（clone_text）如下：
{TEXT}

以下是相关的文件地址说明：
给你提供的音频文件（audio_path）路径为：'./audio_clone/data/新书推荐《AI革命》.wav'。
10s音频文件（output_audio_file）路径为：'./audio_clone/output/10s.wav'，10s音频文字（output_text_file）路径为：'./audio_clone/output/10s.txt'。
生成的语音文件（tts_file_path）路径为：'./audio_clone/output/cloned_voice.mp3'。
""".format(TEXT=TEXT)

for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content=content)]}, config
):
    try:
        # 如果 chunk 是字典，直接序列化
        print(json.dumps(chunk, indent=4, ensure_ascii=False))
    except TypeError:
        # 对于复杂对象，将其转化为字典
        if hasattr(chunk, "__dict__"):
            print(json.dumps(chunk.__dict__, indent=4, ensure_ascii=False))
        else:
            print(f"Unserializable object: {chunk}")
    time.sleep(3)
    print("-----------------------------------------------------------------")
