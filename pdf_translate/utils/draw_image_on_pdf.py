from PyPDF2 import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
import io
import tempfile


def draw_images_on_pdf(input_pdf_path, output_pdf_path, images_data):
    """
    根据提供的图片数据，将图片绘制到指定的 PDF 页面位置。

    :param input_pdf_path: 输入的 PDF 文件路径。
    :param output_pdf_path: 输出的 PDF 文件路径。
    :param images_data: 包含图片路径和绘制信息的列表。
    """
    # 读取原始 PDF 文件
    reader = PdfReader(input_pdf_path)
    writer = PdfWriter()

    for page_num, page in enumerate(reader.pages, start=1):
        # 创建一个空的 PDF 页面
        packet = io.BytesIO()
        can = canvas.Canvas(packet, pagesize=A4)
        page_width, page_height = A4

        # 遍历图片数据，绘制到当前页
        page_images = [img for img in images_data if img["page_num"] == page_num]
        if not page_images:
            print(f"第 {page_num} 页没有图片数据，跳过。")
            writer.add_page(page)
            continue

        for image_info in page_images:
            image_path = image_info["path"]
            bbox = image_info["bbox"]  # [x0, y0, x1, y1]

            # 转换 bbox 坐标到 reportlab 坐标系
            x0, y0, x1, y1 = bbox
            y0, y1 = page_height - y1, page_height - y0  # 转换为从左下角开始

            # 绘制图片
            try:
                can.drawImage(image_path, x0, y0, width=x1 - x0, height=y1 - y0)
            except Exception as e:
                print(f"图片绘制失败: {image_path}, 错误: {e}")

        # 保存新页面内容
        can.save()
        packet.seek(0)

        # 临时保存 PDF 页面
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(packet.getvalue())
            temp_pdf.close()

            # 使用 PyPDF2 合并页面
            new_pdf = PdfReader(temp_pdf.name)
            if len(new_pdf.pages) == 0:
                print(f"警告：生成的 PDF 页面为空，跳过第 {page_num} 页！")
                writer.add_page(page)
                continue

            page.merge_page(new_pdf.pages[0])
        writer.add_page(page)

    # 保存最终合成的 PDF
    with open(output_pdf_path, "wb") as f:
        writer.write(f)

    print(f"图片已成功绘制到 PDF: {output_pdf_path}")
