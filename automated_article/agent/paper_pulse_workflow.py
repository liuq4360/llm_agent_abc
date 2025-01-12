import os
import smtplib
import sys
sys.path.append('./')
from automated_article.utils.get_new_articles_from_arxiv import get_article_for_keywords
from automated_article.utils.download_new_articles import download_all_papers
from automated_article.utils.write_abstracts import write_abstract_api
from automated_article.utils.send_email import send_email
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from typing import List, Dict


# 定义工作流的状态类，继承 Pydantic 的 BaseModel
class WorkflowState(BaseModel):
    keywords: List[str] = Field(default_factory=list, description="用于搜索的关键词列表")
    days: int = Field(1, description="搜索的天数")
    articles: List[Dict] = Field(default_factory=list, description="搜索到的文章列表")
    file_path: str = Field("./automated_article/output/articles/", description="文章保存路径")
    file_info: List[Dict] = Field(default_factory=list, description="已下载文章信息")
    summary_list: List[Dict] = Field(default_factory=list, description="生成的文章摘要列表")
    summaries: str = Field("", description="摘要的 Markdown 格式内容")
    recipient_email: str = Field("", description="接收文章摘要的邮箱地址")


# 节点1：基于关键词搜索 arXiv 文章
def search_arxiv(state):

    def remove_duplicate_links(data):
        # 使用一个集合来跟踪已经见过的链接
        seen_links = set()
        unique_data = []

        for record in data:
            if record['link'] not in seen_links:
                unique_data.append(record)
                seen_links.add(record['link'])  # 记录已见的链接

        return unique_data

    print("🔍 正在搜索与关键词相关的 arXiv 文章...")
    info = {
        "keywords": state.keywords,
        "days": state.days
    }
    articles = get_article_for_keywords(info)
    articles = remove_duplicate_links(articles)
    state.articles = articles
    return state


# 节点2：下载文章到本地
def download_articles(state):
    print("⬇️ 正在下载文章...")
    output_dir = state.file_path
    os.makedirs(output_dir, exist_ok=True)
    links = [article['link'] for article in state.articles]
    download_all_papers(links, output_dir)
    file_list = [f'{link.split("/")[-1]}.pdf' for link in links]
    file_info = [{"file": file, "link": link} for file, link in zip(file_list, links)]
    state.file_info = file_info
    return state


# 节点3：使用大模型生成文章摘要
def generate_summaries(state):
    print("📝 正在生成文章摘要...")
    summary_list = write_abstract_api(state.file_info)
    state.summary_list = summary_list
    return state


# 节点4：发送邮件，推送文章
def push_articles(state):
    print("📧 正在发送邮件...")
    # 生成 markdown 格式的内容
    summary_list = [
        f"{idx + 1}. ** {article['link']} **\n\n{article['summary']}"
        for idx, article in enumerate(state.summary_list)
    ]

    summaries = "\n\n".join(summary_list)
    state.summaries = summaries

    try:
        # 发送邮件
        send_email(
            receiver_email=state.recipient_email,
            subject=f"每日关于关键词{state.keywords}的最新文章摘要",
            content=state.summaries,
        )
    except smtplib.SMTPException as e:
        print(f"发送邮件失败: {e}")
    except Exception as e:
        print(f"其他错误: {e}")

    return state


# 定义工作流图
def build_workflow():
    workflow = StateGraph(WorkflowState)
    workflow.add_node("search_arxiv", search_arxiv)
    workflow.add_node("download_articles", download_articles)
    workflow.add_node("generate_summaries", generate_summaries)
    workflow.add_node("push_articles", push_articles)

    # 定义节点间的执行顺序
    workflow.set_entry_point("search_arxiv")
    workflow.add_edge("search_arxiv", "download_articles")
    workflow.add_edge("download_articles", "generate_summaries")
    workflow.add_edge("generate_summaries", "push_articles")
    workflow.add_edge("push_articles", END)

    return workflow.compile()


# 运行工作流
if __name__ == "__main__":
    # 初始化状态
    initial_state = WorkflowState()
    initial_state.keywords = ['agent']  # 关键词
    initial_state.days = 1  # 搜索天数
    initial_state.recipient_email = "891391257@qq.com"  # 接收者邮箱

    # 编译并运行工作流
    workflow_app = build_workflow()
    print("🚀 开始执行工作流...")
    final_state = workflow_app.invoke(initial_state)
    print("✅ 工作流执行完毕！")
