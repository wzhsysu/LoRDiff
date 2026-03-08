import os
import random
import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from pathlib import Path

import numpy as np
# from src.datasets.realesrgan import RealESRGAN_degradation


def reflect_pad_min_side(img: Image.Image, target_min_size: int) -> Image.Image:
    """
    Reflect pad the input PIL image so that its shortest side equals target_min_size,
    preserving the aspect ratio. No scaling is applied, only padding.

    Args:
        img (PIL.Image): Input image.
        target_min_size (int): Target size for the shortest side.

    Returns:
        PIL.Image: Padded image with preserved aspect ratio.
    """
    w, h = img.size
    min_side = min(w, h)

    if min_side >= target_min_size:
        return img  # no need to pad

    # Calculate needed padding
    pad_h = max(target_min_size - h, 0)
    pad_w = max(target_min_size - w, 0)

    # Pad equally on both sides
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    # Convert to tensor
    img_tensor = TF.to_tensor(img).unsqueeze(0)  # (1, C, H, W)

    # Reflect padding
    img_padded = F.pad(img_tensor, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect')

    # Back to PIL
    img_padded = TF.to_pil_image(img_padded.squeeze(0))

    return img_padded

def resize_min_side(img: Image.Image, target_min_size: int) -> Image.Image:
    """
    Resize the input PIL image so that its shortest side equals target_min_size,
    preserving the aspect ratio. Uses bicubic interpolation.

    Args:
        img (PIL.Image): Input image.
        target_min_size (int): Target size for the shortest side.

    Returns:
        PIL.Image: Resized image with preserved aspect ratio.
    """
    w, h = img.size
    min_side = min(w, h)

    if min_side >= target_min_size:
        return img  # no need to resize

    # Compute scaling factor
    scale = target_min_size / min_side
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    # Resize using bicubic interpolation
    return img.resize((new_w, new_h), Image.BICUBIC)

class SYSTxtDataset(torch.utils.data.Dataset):
    def __init__(self, args=None):
        super().__init__()

        self.args = args
        self.crop_preproc = transforms.Compose([
            transforms.RandomCrop((args.resolution_ori, args.resolution_ori)),
            transforms.Resize((args.resolution_tgt, args.resolution_tgt)),
            transforms.RandomHorizontalFlip(),
        ])
        self.to_tensor = transforms.ToTensor()
        
        with open(args.dataset_txt_paths, 'r') as f:
            self.gt_list = [line.strip() for line in f.readlines()]
        if args.highquality_dataset_txt_paths is not None:
            with open(args.highquality_dataset_txt_paths, 'r') as f:
                self.hq_gt_list = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.gt_list)

    def __getitem__(self, idx):

        if self.args.highquality_dataset_txt_paths is not None:
            if np.random.uniform() < self.args.prob:
                gt_img = Image.open(self.gt_list[idx]).convert('RGB')
            else:
                idx = random.sample(range(0, len(self.hq_gt_list)), 1)
                gt_img = Image.open(self.hq_gt_list[idx[0]]).convert('RGB')
        else:
            gt_img = Image.open(self.gt_list[idx]).convert('RGB')

        gt_img = resize_min_side(gt_img, target_min_size=512)
        gt_img = self.crop_preproc(gt_img)

        output_t = self.to_tensor(gt_img)#, img_t = self.degradation.degrade_process(np.asarray(gt_img)/255., resize_bak=True)
        #output_t, img_t = output_t.squeeze(0), img_t.squeeze(0)
        # output images scaled to -1,1
        output_t = (output_t -0.5)/0.5
        example = {}
        # example["prompt"] = caption
        example["neg_prompt"] = self.args.neg_prompt_csd
        example["null_prompt"] = ""
        example["output_pixel_values"] = output_t
        # example["conditioning_pixel_values"] = img_t
        return example
       

class PairedLLDataset(torch.utils.data.Dataset):
    def __init__(self, folder, args=None):
        super().__init__()
        self.args = args
        self.input_folder = os.path.join(folder, "Low")
        self.output_folder = os.path.join(folder, "Normal")
        
        lr_names = sorted(os.listdir(self.input_folder))
        gt_names = sorted(os.listdir(self.output_folder))
        common_names = sorted(set(lr_names) & set(gt_names))

        self.lr_list = [os.path.join(self.input_folder, name) for name in common_names]
        self.gt_list = [os.path.join(self.output_folder, name) for name in common_names]

        assert len(self.lr_list) == len(self.gt_list)
        
        self.res_ori = args.resolution_ori
        self.res_tgt = args.resolution_tgt

    def __len__(self):
        return len(self.gt_list)

    def __getitem__(self, idx):
        input_img = Image.open(self.lr_list[idx]).convert('RGB')
        output_img = Image.open(self.gt_list[idx]).convert('RGB')
        
        input_img = reflect_pad_min_side(input_img, target_min_size=512)
        output_img = reflect_pad_min_side(output_img, target_min_size=512)
        
        # 同步crop
        i, j, h, w = transforms.RandomCrop.get_params(input_img, output_size=(self.res_ori, self.res_ori))
        input_img = TF.resized_crop(input_img, i, j, h, w, size=(self.res_tgt, self.res_tgt))
        output_img = TF.resized_crop(output_img, i, j, h, w, size=(self.res_tgt, self.res_tgt))
        
        # input images scaled to [-1, 1]
        input_t = (TF.to_tensor(input_img) - 0.5) / 0.5
        output_t = (TF.to_tensor(output_img) - 0.5) / 0.5

        return {
            "neg_prompt": self.args.neg_prompt_csd,
            "null_prompt": "",
            "output_pixel_values": output_t,
            "conditioning_pixel_values": input_t,
            "base_name": os.path.basename(self.lr_list[idx])
        }



# import os
# from torch.utils.data import DataLoader
# from torchvision.utils import save_image
# from tqdm import tqdm

# # 导入你上面的代码，这里假设在 `dataset.py` 文件中
# # from dataset import PairedLLDataset

# def test_paired_loader(folder, args, save_dir, num_samples=10):
#     """
#     测试 PairedLLDataset，将图片保存到指定文件夹。

#     Args:
#         folder (str): 数据集所在的根目录。
#         args (argparse.Namespace): 参数配置。
#         save_dir (str): 保存图片的文件夹路径。
#         num_samples (int): 要保存的样本数。
#     """
#     os.makedirs(save_dir, exist_ok=True)

#     # 创建数据集和DataLoader
#     dataset = PairedLLDataset(folder, args)
#     dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

#     for idx, batch in enumerate(tqdm(dataloader)):
#         # 只保存指定数量
#         if idx >= num_samples:
#             break

#         # 还原到 [0, 1] 方便保存
#         input_img = (batch['conditioning_pixel_values'][0] * 0.5 + 0.5).clamp(0, 1)
#         output_img = (batch['output_pixel_values'][0] * 0.5 + 0.5).clamp(0, 1)

#         base_name = batch['base_name'][0]

#         # 保存文件
#         input_save_path = os.path.join(save_dir, f"{base_name}_input.png")
#         output_save_path = os.path.join(save_dir, f"{base_name}_output.png")

#         save_image(input_img, input_save_path)
#         save_image(output_img, output_save_path)

#     print(f"已保存 {min(num_samples, len(dataset))} 对图像到 {save_dir}")

# # 示例 args
# class Args:
#     def __init__(self):
#         self.resolution_ori = 512  # 原始裁剪分辨率
#         self.resolution_tgt = 256  # 最终resize到的分辨率
#         self.neg_prompt_csd = "negative prompt example"

# # 使用示例
# if __name__ == '__main__':
#     data_folder = '/data2/zhihua/LLIE/dataset/LOLTrain/Train'  # 数据集路径
#     output_folder = './test'  # 保存结果的路径
#     args = Args()
#     test_paired_loader(data_folder, args, output_folder, num_samples=20)
