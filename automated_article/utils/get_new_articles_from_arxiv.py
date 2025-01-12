import requests
from bs4 import BeautifulSoup
from typing import Dict, List
from datetime import datetime, timedelta
import sys
sys.path.append('./')
from automated_article.configs.model_config import ARXIV_BASE_URL


def search_arxiv_advanced(base_url, params):
    """
    使用 arXiv 高级搜索功能获取文章信息。

    :param base_url: 高级搜索 URL
    :param params: 搜索参数（包含关键词、日期范围等）
    :return: 包含文章信息的列表
    """
    # 请求页面
    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data from arXiv: {response.status_code}")

    # 解析 HTML 内容
    soup = BeautifulSoup(response.text, 'html.parser')
    results = []

    # 查找搜索结果列表
    articles = soup.find_all('li', class_='arxiv-result')
    for article in articles:
        title_tag = article.find('p', class_='title is-5 mathjax')
        abstract_tag = article.find('span', class_='abstract-full has-text-grey-dark mathjax')
        authors_tag = article.find('p', class_='authors')
        date_tag = article.find('p', class_='is-size-7')

        title = title_tag.text.strip() if title_tag else "N/A"
        abstract = abstract_tag.text.strip() if abstract_tag else "N/A"
        authors = authors_tag.text.strip() if authors_tag else "N/A"
        date = date_tag.text.strip() if date_tag else "N/A"

        # 提取文章链接
        link_tag = article.find('p', class_='list-title is-inline-block')
        link = link_tag.find('a')['href'] if link_tag and link_tag.find('a') else "N/A"

        results.append({
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "date": date,
            "link": link
        })

    return results


def get_article_for_keywords(info: Dict) -> List[Dict]:
    keywords = info['keywords']
    days = info['days']

    # 获取昨天日期
    yesterday = datetime.now() - timedelta(days=days)

    # 格式化为字符串
    formatted_yesterday = yesterday.strftime("%Y-%m-%d")

    # 获取当天日期
    today = datetime.now()
    # 格式化为字符串
    formatted_today = today.strftime("%Y-%m-%d")  # 输出格式：2024-11-26

    start_time = info.get('start_time', formatted_yesterday)
    end_time = info.get('end_time', formatted_today)
    size = info.get('size', 25)
    if not keywords:
        return []
    articles = []
    for keyword in keywords:
        params = {
            "advanced": "",
            "terms-0-operator": "AND",
            "terms-0-term": keyword,
            "terms-0-field": "title",
            "terms-1-operator": "AND",
            "terms-1-term": keyword,
            "terms-1-field": "abstract",
            "classification-computer_science": "y",
            "classification-include_cross_list": "include",
            "date-filter_by": "date_range",
            "date-from_date": start_time,
            "date-to_date": end_time,
            "date-date_type": "submitted_date",
            "abstracts": "show",
            "size": size,
            "order": "-announced_date_first"
        }
        articles.extend(search_arxiv_advanced(ARXIV_BASE_URL, params))

    if articles:
        print(f"With keywords {keywords}, Found {len(articles)} articles.")
        return articles
    else:
        return []


if __name__ == "__main__":

    info = {
        "keywords": ["agent"],
        "days": 1
    }
    articles = get_article_for_keywords(info)
    print(f"Found {len(articles)} articles:")
    for article in articles:
        print(f"- Title: {article['title']}")
        print(f"  Authors: {article['authors']}")
        print(f"  Abstract: {article['abstract'][:200]}...")  # 截断摘要方便查看
        print(f"  Link: {article['link']}")
        print(f"  Date: {article['date']}\n")
