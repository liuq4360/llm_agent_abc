import cv2
import easyocr
import numpy as np
import sys
sys.path.append('./')
from utils import read_json_file
import math
from torch import nn
import torch
import PIL.Image as pil_image


def detect_and_remove_edge_text(image_path, output_path, margin=50):
    """
    检测并移除图片边缘的文字。

    :param image_path: 输入图片路径
    :param output_path: 输出图片路径
    :param margin: 判断文字是否在边缘的边距值（像素）
    """
    # 读取图像
    image = cv2.imread(image_path)
    h, w, _ = image.shape

    # 使用 EasyOCR 检测文字
    reader = easyocr.Reader(['en', 'ch_sim'])  # 英文、简体中文
    detections = reader.readtext(image)

    # 创建空白掩码
    mask = np.zeros((h, w), dtype=np.uint8)

    # 根据文字区域绘制掩码，仅保留边缘的文字区域
    for detection in detections:
        box = detection[0]
        pts = np.array(box, dtype=np.int32)

        # 检查文字区域是否接近边缘
        x_min, y_min = np.min(pts, axis=0)
        x_max, y_max = np.max(pts, axis=0)

        if (x_min < margin or x_max > w - margin or
                y_min < margin or y_max > h - margin):
            cv2.fillPoly(mask, [pts], 255)

    # 修复图像
    inpainted_image = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    # 保存结果
    cv2.imwrite(output_path, inpainted_image)


class FSRCNN(nn.Module):
    def __init__(self, scale_factor, num_channels=1, d=56, s=12, m=4):
        super(FSRCNN, self).__init__()
        self.first_part = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=5, padding=5//2),
            nn.PReLU(d)
        )
        self.mid_part = [nn.Conv2d(d, s, kernel_size=1), nn.PReLU(s)]
        for _ in range(m):
            self.mid_part.extend([nn.Conv2d(s, s, kernel_size=3, padding=3//2), nn.PReLU(s)])
        self.mid_part.extend([nn.Conv2d(s, d, kernel_size=1), nn.PReLU(d)])
        self.mid_part = nn.Sequential(*self.mid_part)
        self.last_part = nn.ConvTranspose2d(d, num_channels, kernel_size=9, stride=scale_factor, padding=9//2,
                                            output_padding=scale_factor-1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.first_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        for m in self.mid_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        nn.init.normal_(self.last_part.weight.data, mean=0.0, std=0.001)
        nn.init.zeros_(self.last_part.bias.data)

    def forward(self, x):
        x = self.first_part(x)
        x = self.mid_part(x)
        x = self.last_part(x)
        return x


def convert_ycbcr_to_rgb(img, dim_order='hwc'):
    if dim_order == 'hwc':
        r = 298.082 * img[..., 0] / 256. + 408.583 * img[..., 2] / 256. - 222.921
        g = 298.082 * img[..., 0] / 256. - 100.291 * img[..., 1] / 256. - 208.120 * img[..., 2] / 256. + 135.576
        b = 298.082 * img[..., 0] / 256. + 516.412 * img[..., 1] / 256. - 276.836
    else:
        r = 298.082 * img[0] / 256. + 408.583 * img[2] / 256. - 222.921
        g = 298.082 * img[0] / 256. - 100.291 * img[1] / 256. - 208.120 * img[2] / 256. + 135.576
        b = 298.082 * img[0] / 256. + 516.412 * img[1] / 256. - 276.836
    return np.array([r, g, b]).transpose([1, 2, 0])


def convert_rgb_to_ycbcr(img, dim_order='hwc'):
    if dim_order == 'hwc':
        y = 16. + (64.738 * img[..., 0] + 129.057 * img[..., 1] + 25.064 * img[..., 2]) / 256.
        cb = 128. + (-37.945 * img[..., 0] - 74.494 * img[..., 1] + 112.439 * img[..., 2]) / 256.
        cr = 128. + (112.439 * img[..., 0] - 94.154 * img[..., 1] - 18.285 * img[..., 2]) / 256.
    else:
        y = 16. + (64.738 * img[0] + 129.057 * img[1] + 25.064 * img[2]) / 256.
        cb = 128. + (-37.945 * img[0] - 74.494 * img[1] + 112.439 * img[2]) / 256.
        cr = 128. + (112.439 * img[0] - 94.154 * img[1] - 18.285 * img[2]) / 256.
    return np.array([y, cb, cr]).transpose([1, 2, 0])


def preprocess(img, device):
    img = np.array(img).astype(np.float32)
    ycbcr = convert_rgb_to_ycbcr(img)
    x = ycbcr[..., 0]
    x /= 255.
    x = torch.from_numpy(x).to(device)
    x = x.unsqueeze(0).unsqueeze(0)
    return x, ycbcr


def fsrcnn_super_resolution(input_image_path, output_image_path, model_path, upscale_factor=3):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = FSRCNN(scale_factor=upscale_factor).to(device)

    state_dict = model.state_dict()
    for n, p in torch.load(model_path, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model.eval()

    image = pil_image.open(input_image_path).convert('RGB')

    image_width = (image.width // upscale_factor) * upscale_factor
    image_height = (image.height // upscale_factor) * upscale_factor

    hr = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
    lr = hr.resize((hr.width // upscale_factor, hr.height // upscale_factor), resample=pil_image.BICUBIC)
    bicubic = lr.resize((lr.width * upscale_factor, lr.height * upscale_factor), resample=pil_image.BICUBIC)

    lr, _ = preprocess(lr, device)
    hr, _ = preprocess(hr, device)
    _, ycbcr = preprocess(bicubic, device)

    with torch.no_grad():
        preds = model(lr).clamp(0.0, 1.0)

    preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

    output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
    output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
    output = pil_image.fromarray(output)

    # 保存输出图像
    output.save(output_image_path)
    print(f"超分辨率图片已保存到: {output_image_path}")


if __name__ == "__main__":
    candidate_poster_dir = 'video_poster_generate/output/candidate_poster.json'
    candidate_poster = read_json_file(candidate_poster_dir)

    image_dir = candidate_poster['image_dir']

    poster_dir = 'video_poster_generate/output/poster_without_text.jpg'

    detect_and_remove_edge_text(image_dir, poster_dir)

    output_image = "video_poster_generate/output/final_poster.jpg"

    model_path = "video_poster_generate/models/fsrcnn_x3.pth"  # 预训练 FSRCNN 模型路径

    # 调用函数
    fsrcnn_super_resolution(poster_dir, output_image, model_path, upscale_factor=3)
