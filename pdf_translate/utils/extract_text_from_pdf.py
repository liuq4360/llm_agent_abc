import pdfplumber


def get_paragraph_lines_info(paragraph_lines: list[dict]) -> dict:
    """
    提取段落的相关信息：合并文本，获取段落边界框（top, bottom, x0, x1）。

    :param paragraph_lines:
        [
                {
                    "text": "Given the lack of standardized datasets specific to chip design, we adopted the ChipNeMo [90]",
                    "top": 194,
                    "bottom": 205,
                    "x0": 72.0,
                    "x1": 540.0045790943997
                },
                ...
        ]
    :return:
        {
            "text": "Combined text from all lines",
            "top": 194,
            "bottom": 246,
            "x0": 72.0,
            "x1": 540.0045790943997
        }
    """
    # 检查每一行是否包含 'bottom' 和其他关键字段
    for line in paragraph_lines:
        if 'bottom' not in line:
            line['bottom'] = line.get('top', 0)  # 如果没有bottom字段，则使用top字段值进行补充

    text_list = [line['text'] for line in paragraph_lines]
    text = " ".join(text_list)
    top = paragraph_lines[0]['top']
    bottom = paragraph_lines[-1]['bottom']  # 获取最后一行的bottom值
    x0_list = [line['x0'] for line in paragraph_lines]
    x1_list = [line['x1'] for line in paragraph_lines]
    x0 = min(x0_list)
    x1 = max(x1_list)

    return {
        "text": text,
        "top": top,
        "bottom": bottom,
        "x0": x0,
        "x1": x1
    }


def extract_text_by_paragraph(pdf_path, page_width_threshold=0.85, line_width_difference_ratio=0.05, tolerance=3):
    """
    基于行的分段和合并策略，按段落提取 PDF 内容。

    Args:
        pdf_path (str): PDF 文件路径。
        page_width_threshold (float): 页面宽度占比阈值（0~1），超过该比例的行视为段落。

    Returns:
        list: 包含每页段落的列表，每个段落包含文字及排版信息。
        :param line_width_difference_ratio:
    """

    extracted_data = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            page_width = page.width  # 当前页的总宽度
            lines = []

            # 第一步：按行提取文字
            for word_info in page.extract_words():
                # 将同一行的文字合并
                top = round(word_info["top"])  # 取整数以避免浮点误差
                bottom = round(word_info["bottom"])
                if not lines or abs(lines[-1]["top"] - top) > tolerance:
                    # 如果是新的一行，创建新的行数据
                    lines.append({
                        "text": word_info["text"],
                        "top": top,
                        "bottom": bottom,
                        "x0": word_info["x0"],
                        "x1": word_info["x1"],
                    })
                else:
                    # 同一行的文字合并
                    lines[-1]["text"] += f" {word_info['text']}"
                    lines[-1]["x0"] = min(lines[-1]["x0"], word_info["x0"])  # 更新行的左边界
                    lines[-1]["x1"] = max(lines[-1]["x1"], word_info["x1"])  # 更新行的右边界

            # print(json.dumps(lines, indent=4, ensure_ascii=False))

            for word_info in page.extract_words():
                # 跳过表格中的文字
                if page.find_tables():
                    for table in page.find_tables():
                        if table.bbox[1] <= word_info["top"] <= table.bbox[3]:
                            break  # 属于表格内容，跳过
                    else:
                        # 非表格内容，合并行文字
                        top = round(word_info["top"])
                        if not lines or lines[-1]["top"] != top:
                            lines.append({
                                "text": word_info["text"],
                                "top": top,
                                "x0": word_info["x0"],
                                "x1": word_info["x1"]
                            })
                        else:
                            lines[-1]["text"] += f" {word_info['text']}"

            # 第二步：按段落合并行
            paragraphs = []
            paragraph_lines = []
            for i, line in enumerate(lines):
                is_last_line = i == len(lines) - 1  # 判断是否是最后一行
                line_width_ratio = line["x1"] / page_width  # 当前行宽度占比

                if i == 0:  # 第一行
                    # 段落的起始
                    paragraph_lines.append(line)
                    if (
                            not is_last_line
                            and abs(lines[i + 1]["x1"] - line["x1"]) / line["x1"] > line_width_difference_ratio
                    ):  # 段落结束
                        paragraph = {"page_num": page_num + 1}
                        paragraph.update(get_paragraph_lines_info(paragraph_lines))
                        paragraphs.append(paragraph)
                        paragraph_lines = []
                        continue
                elif (  # 中间行
                        i > 0 and
                        not is_last_line
                ):
                    if (line_width_ratio > page_width_threshold
                            and abs(lines[i - 1]["x1"] - line["x1"]) / line["x1"] > line_width_difference_ratio):
                        # 段落的起始
                        paragraph_lines = [line]
                        if (
                                not is_last_line
                                and abs(lines[i + 1]["x1"] - line["x1"]) / line["x1"] > line_width_difference_ratio
                        ):  # 段落结束
                            paragraph = {"page_num": page_num + 1}
                            paragraph.update(get_paragraph_lines_info(paragraph_lines))
                            paragraphs.append(paragraph)
                            paragraph_lines = []
                            continue
                    elif (
                            not is_last_line
                            and line_width_ratio > page_width_threshold
                            and abs(lines[i - 1]["x1"] - line["x1"]) / line["x1"] < line_width_difference_ratio
                    ):
                        # 当前行是段落的中间部分，合并到段落中
                        paragraph_lines.append(line)
                    else:
                        # 当前行结束段落
                        paragraph_lines.append(line)
                        paragraph = {"page_num": page_num + 1}
                        paragraph.update(get_paragraph_lines_info(paragraph_lines))
                        paragraphs.append(paragraph)
                        paragraph_lines = []
                elif is_last_line:  # 末尾行
                    paragraph_lines = [line]
                    paragraph = {"page_num": page_num + 1}
                    paragraph.update(get_paragraph_lines_info(paragraph_lines))
                    paragraphs.append(paragraph)

            extracted_data.append(paragraphs)

    return extracted_data
