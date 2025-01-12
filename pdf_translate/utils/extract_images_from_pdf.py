import fitz  # PyMuPDF
import os

"""
针对抽取的图片中的字段进行说明。针对下面这个案例说明：
{
    "page_num": 4,
    "image_index": 1,
    "path": "./pdf_translate/output/images/page_4_img_1.png",
    "image_ext": "png",
    "width": 3747,
    "height": 1520,
    "colorspace": 3,
    "bbox": [
        93.54499816894531,
        471.3414306640625,
        501.7056884765625,
        636.9150390625
    ]
}

1.	page_num:
	•	含义: 图片所在的 PDF 页码（从 1 开始计数）。
	•	单位: 无单位，整数值。
	•	例子: "page_num": 4 表示该图片位于 PDF 文件的第 4 页。
	2.	image_index:
	•	含义: 该图片在页面中的索引。每页上的图片都有一个索引值，用于标识该页中的图片顺序。
	•	单位: 无单位，整数值。
	•	例子: "image_index": 1 表示这是该页面上的第一张图片。
	3.	path:
	•	含义: 提取的图片文件保存路径。该路径指向提取并保存的图片文件。
	•	单位: 无单位，字符串表示文件路径。
	•	例子: "path": "./pdf_translate/output/images/page_4_img_1.png" 表示该图片保存在 ./pdf_translate/output/images/ 目录下，文件名为 page_4_img_1.png。
	4.	image_ext:
	•	含义: 提取图片的文件扩展名（即图片的格式）。如 png、jpg、jpeg 等。
	•	单位: 无单位，字符串表示文件扩展名。
	•	例子: "image_ext": "png" 表示该图片被保存为 PNG 格式。
	5.	width:
	•	含义: 图片的宽度，单位是 像素 (px)。这是提取图片的实际分辨率，表示图片的水平像素数量。
	•	单位: 像素（px）。
	•	例子: "width": 3747 表示图片的宽度为 3747 像素。
	6.	height:
	•	含义: 图片的高度，单位是 像素 (px)。这是提取图片的实际分辨率，表示图片的垂直像素数量。
	•	单位: 像素（px）。
	•	例子: "height": 1520 表示图片的高度为 1520 像素。
	7.	colorspace:
	•	含义: 图片的颜色空间。不同的颜色空间编码方式定义了图片颜色的表现方式。
	•	1 表示 灰度（Grayscale），通常是单色图像。
	•	3 表示 RGB 颜色空间（Red, Green, Blue），常见于彩色图片。
	•	其他值也代表不同的颜色空间（如 CMYK）。
	•	单位: 无单位，整数值，代表颜色空间的编码方式。
	•	例子: "colorspace": 3 表示该图片使用 RGB 颜色空间。
	8.	bbox:
	•	含义: 图片在 PDF 页面中的边界框（bounding box），定义了图片在页面中的位置和尺寸。
	•	[x0, y0, x1, y1]：x0, y0 表示图片的左下角坐标，x1, y1 表示图片的右上角坐标。
	•	这些值是基于 PDF 页面坐标系，单位是 点 (pt)，PDF 的标准单位。1 点 = 1/72 英寸。
	•	这个边界框定义了图片在页面中的位置和图片的尺寸。图片的显示区域由此边界框决定。
	•	单位: 点（pt），PDF 的标准坐标单位。
	•	例子:
	
	"bbox": [
        93.54499816894531,
        471.3414306640625,
        501.7056884765625,
        636.9150390625
    ]
    
    •	93.54499816894531 是图片左下角的 x 坐标。
	•	471.3414306640625 是图片左下角的 y 坐标。
	•	501.7056884765625 是图片右上角的 x 坐标。
	•	636.9150390625 是图片右上角的 y 坐标。
	
	
	单位说明

	•	像素 (px):
	•	用于 width 和 height 字段，表示图片的分辨率。像素是图像显示的基本单位，定义了图像的精度和清晰度。
	•	点 (pt):
	•	用于 bbox 字段，表示图片在 PDF 页面中的坐标位置。PDF 坐标系统以点为单位，1 点等于 1/72 英寸。bbox 提供了图片在 PDF 页面中的具体位置和大小。
	
"""


def extract_images(pdf_path, output_dir):
    """
    从 PDF 提取嵌入图片，并返回正确的边界框 (bbox)。

    Args:
        pdf_path (str): PDF 文件路径。
        output_dir (str): 保存图片的目录。

    Returns:
        list: 包含图片路径、分辨率、颜色空间和位置的元数据列表。
    """
    os.makedirs(output_dir, exist_ok=True)
    image_metadata = []

    pdf_document = fitz.open(pdf_path)

    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        images = page.get_images(full=True)

        # 遍历页面中的每一张图片
        for img_index, img in enumerate(images):
            xref = img[0]  # 图片引用号
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            # 提取图片的 bbox
            bbox = None
            for img_info in page.get_images(full=True):
                if img_info[0] == xref:
                    bbox = page.get_image_bbox(img_info)

            # 保存图片
            image_filename = f"page_{page_num + 1}_img_{img_index + 1}.{image_ext}"
            image_path = os.path.join(output_dir, image_filename)
            with open(image_path, "wb") as image_file:
                image_file.write(image_bytes)

            # 记录图像元数据
            image_metadata.append({
                "page_num": page_num + 1,
                "image_index": img_index + 1,
                "path": image_path,
                "image_ext": image_ext,
                "width": base_image["width"],
                "height": base_image["height"],
                "colorspace": base_image["colorspace"],
                "bbox": list(bbox) if bbox else None,  # 确保 bbox 是列表格式
            })

    pdf_document.close()
    return image_metadata
