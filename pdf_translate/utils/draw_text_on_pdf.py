import io
from reportlab.pdfgen import canvas
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics


# 注册中文字体（替换成实际路径）
pdfmetrics.registerFont(TTFont("Songti", "/System/Library/Fonts/Supplemental/Songti.ttc"))  # 替换为宋体路径


def get_page_min_font_size(page_data, adjusted_width, page_height, margin=20, min_font_size=8, max_font_size=12):
    """
    计算一页内所有段落的最小字体大小，确保段落高度不超出页面范围。

    :param page_data: 当前页的段落数据。
    :param adjusted_width: 可用宽度。
    :param page_height: 页面高度。
    :param margin: 页边距。
    :param min_font_size: 最小字体大小。
    :param max_font_size: 最大字体大小。
    :return: 适配该页所有段落的最小字体大小。
    """
    min_size = max_font_size
    max_height = page_height - 2 * margin  # 可用的最大高度

    for item in page_data:
        text = item["text"]
        top = item["top"]
        bottom = item["bottom"]
        height = min(bottom - top, max_height)  # 限制高度为页面内可用范围

        try:
            font_size, _, _, _ = calculate_font_size_and_layout(
                text, adjusted_width, height, min_font_size
            )
            min_size = min(min_size, font_size)  # 获取最小字体大小
        except ValueError:
            continue  # 跳过无法容纳的段落

    return max(min_size, min_font_size)  # 不低于最小字体大小


def generate_pdf_with_text(translated_data, output_path, margin=20, min_font_size=8, max_font_size=12):
    """
    使用 reportlab 按原排版生成翻译后的 PDF，段落从 x0 开始，确保右边距。
    :param translated_data: 翻译后的数据，按页组织。
    :param output_path: 输出 PDF 文件路径。
    :param margin: 页面的右边距，默认 20 像素。
    :param min_font_size: 最小字体大小。
    :param max_font_size: 最大字体大小。
    """

    packet = io.BytesIO()
    c = canvas.Canvas(packet, pagesize=A4)
    page_width = A4[0]
    page_height = A4[1]

    for page_data in translated_data:
        # 计算当前页的最小字体大小
        page_font_size = get_page_min_font_size(page_data, page_width - 2 * margin, page_height, margin, min_font_size, max_font_size)

        for item in page_data:
            text = item["text"]
            x0 = item["x0"]
            top = item["top"]
            bottom = item["bottom"]
            width = item["x1"] - x0  # 段落宽度
            height = bottom - top

            # 跳过无效段落
            if width <= 0 or page_font_size <= 0:
                print(f"段落宽度或字体大小无效，跳过：width={width}, font_size={page_font_size}")
                continue

            if not text.strip():
                print("段落内容为空，跳过")
                continue

            # 设置字体并检查高度
            c.setFont("Songti", page_font_size)
            if height < page_font_size * 2:  # 如果高度不足，调整为最小高度
                height = page_font_size * 2

            try:
                # 根据段落宽度和字体大小计算排版
                char_width = page_font_size * 1.1
                if text.isdigit() or text.isalpha():  # 数字或字母，按照文本自身的长度来展示，避免页码等数字展示不全。
                    chars_per_line = len(text)
                else:
                    chars_per_line = max(int(width // char_width), 1)  # 至少为 1
                line_height = page_font_size * 2
                total_lines = min(int(height // line_height), len(text) // chars_per_line + 1)

                y = page_height - top  # 起始位置（从页面顶部开始绘制）

                # 绘制文字
                for i in range(total_lines):
                    start = i * chars_per_line
                    end = start + chars_per_line
                    line = text[start:end]

                    if not line.strip():  # 确保内容非空
                        continue

                    c.drawString(x0, y, line)
                    y -= line_height  # 按行高度调整

            except ValueError as e:
                print(f"段落处理失败: {e}")

        c.showPage()

    c.save()
    with open(output_path, "wb") as f:
        f.write(packet.getvalue())


def calculate_font_size_and_layout(text, width, height, min_font_size=4):
    """
    根据段落宽度、高度及文字内容，计算字体大小、每行汉字数量及总行数。

    如果无法完全适配高度，会截断内容。

    :param text: 段落内容。
    :param width: 段落宽度。
    :param height: 段落高度。
    :param min_font_size: 最小字体大小，默认为 4。
    :return: 字体大小、每行汉字数量、总行数、截断后的内容。
    """
    import math

    text_length = len(text)  # 总字符数
    original_text = text  # 保存原始内容
    font_size = 12  # 初始字体大小

    while font_size >= min_font_size:
        char_width = font_size * 1.2  # 字符宽度
        chars_per_line = max(int(width // char_width), 1)

        line_height = font_size * 2  # 行高
        total_lines = math.ceil(text_length / chars_per_line)

        # 如果段落总高度足够容纳所有行，返回计算结果
        if total_lines * line_height <= height:
            return font_size, chars_per_line, total_lines, text

        # 缩小字体继续尝试
        font_size -= 0.5

    # 字体大小达到最小仍无法容纳，截断内容
    max_lines = int(height // (min_font_size * 1.5))  # 最大可容纳行数
    max_chars = max_lines * max(int(width // (min_font_size * 1.1)), 1)  # 最大可容纳字符数

    truncated_text = original_text[:max_chars] + "..."  # 截断并添加省略号
    return min_font_size, max(int(width // (min_font_size * 1.1)), 1), max_lines, truncated_text
