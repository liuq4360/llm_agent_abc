import os
import requests
from tqdm import tqdm


def download_arxiv_paper(link, output_dir="./automated_article/output/articles"):
    """
    下载 arXiv 文章的 PDF 文件到指定目录。

    :param link: arXiv 的文章链接 (如 https://arxiv.org/abs/2411.13543)
    :param output_dir: 保存 PDF 文件的目录
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 将 "abs" 替换为 "pdf" 以获取 PDF 文件地址
    pdf_url = link.replace("abs", "pdf")

    # 提取 arXiv ID 作为文件名
    arxiv_id = link.split("/")[-1]
    output_path = os.path.join(output_dir, f"{arxiv_id}.pdf")
    print(output_path)
    if os.path.exists(output_path):
        print(f"{arxiv_id} already exists. Skipping download.")
        return None
    else:
        try:
            print(f"Downloading {arxiv_id} from {pdf_url}...")
            response = requests.get(pdf_url, stream=True)
            response.raise_for_status()  # 检查 HTTP 请求是否成功

            # 以流的方式保存 PDF 文件
            with open(output_path, "wb") as pdf_file:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        pdf_file.write(chunk)

            print(f"Downloaded: {output_path}")
            return output_path
        except requests.exceptions.RequestException as e:
            print(f"Failed to download {arxiv_id}: {e}")
            return None


def download_all_papers(links: list[str], output_dir="./automated_article/output/articles"):
    """
    下载所有给定链接对应的 arXiv 文章。

    :param links: arXiv 文章链接列表
    :param output_dir: 保存 PDF 文件的目录
    """
    for link in tqdm(links, desc="Downloading papers"):
        download_arxiv_paper(link, output_dir)


if __name__ == "__main__":
    # 示例链接列表（可替换为实际链接）
    paper_links = [
        "https://arxiv.org/abs/2411.14214",
        "https://arxiv.org/abs/2411.13768",
        "https://arxiv.org/abs/2411.13543",
        # 添加更多链接...
    ]
    # 下载所有论文到 ./data 目录
    download_all_papers(paper_links, output_dir="./automated_article/output/articles")
