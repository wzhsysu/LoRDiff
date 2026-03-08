import os
import argparse
import numpy as np
from PIL import Image, ImageOps
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from pisasr1 import PiSASR_eval
import glob
from pathlib import Path

# Calculate required padding
def get_padding(length):
    # Calculate how much padding is needed to make length a multiple of 8
    remainder = length % 256
    if remainder == 0:
        return 0, 0  # no padding needed
    padding = 256 - remainder
    # Split padding into left/right or top/bottom (making it as symmetrical as possible)
    pad_before = padding // 2
    pad_after = padding - pad_before
    return pad_before, pad_after

def reflective_pad(image, left, top, right, bottom):
    # Convert to numpy array (H x W x C)
    np_img = np.array(image)
    
    # Apply reflective padding (corrected axis order)
    padded_img = np.pad(
        np_img,
        ((top, bottom), (left, right), (0, 0)),  # (H, W, C) padding
        mode='reflect'
    )
    return Image.fromarray(padded_img)

def resize_longest_side(img, output_size=1024):
    # Open and convert to RGB
    width, height = img.size

    # Calculate new dimensions
    if width > height:
        new_width = output_size
        new_height = int(height * (output_size / width))
    else:
        new_height = output_size
        new_width = int(width * (output_size / height))

    # Resize with high-quality downsampling
    resized_img = img.resize((new_width, new_height), Image.BICUBIC)
    return resized_img

def img_enlarge(img):
     # Open and convert to RGB
    width, height = img.size
    # Resize with high-quality downsampling
    resized_img = img.resize((width*2, height*2), Image.BICUBIC)
    return resized_img

def img_shrink(img):
     # Open and convert to RGB
    width, height = img.size
    # Resize with high-quality downsampling
    resized_img = img.resize((int(width/2), int(height/2)), Image.BICUBIC)
    return resized_img


def pisa_sr(args):
    # Initialize the model
    model = PiSASR_eval(args)
    model.set_eval()

    # Get all input images
    if os.path.isdir(args.input_image):
        exts = {'.jpg', '.png', '.bmp'}          # file types we care about
        image_names = sorted(
            str(p)                               # turn Path → string if you need plain paths
            for p in Path(args.input_image).iterdir()
            if p.suffix.lower() in exts          # keep only the desired extensions
        )
    else:
        image_names = [args.input_image]

    # Make the output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f'There are {len(image_names)} images.')

    time_records = []
    for image_name in image_names:
        # Ensure the input image is a multiple of 8
        input_image = Image.open(image_name).convert('RGB')
        # print(image_name, input_image.size)
        if args.resize:
            input_image = img_enlarge(input_image)
        
        ori_width, ori_height = input_image.size
        
        pad_left, pad_right = get_padding(ori_width)
        pad_top, pad_bottom = get_padding(ori_height)
        input_image = reflective_pad(input_image, pad_left, pad_top, pad_right, pad_bottom) 
        print(image_name, input_image.size)

        bname = os.path.basename(image_name)
        # Get caption (you can add the text prompt here)
        validation_prompt = ''
        # Translate the image
        with torch.no_grad():
            c_t = F.to_tensor(input_image).unsqueeze(0).cuda() * 2 - 1
            inference_time, output_image = model(args.default, c_t, prompt=validation_prompt)

        print(f"Inference time: {inference_time:.4f} seconds")
        time_records.append(inference_time)

        output_image = output_image * 0.5 + 0.5
        output_image = torch.clip(output_image, 0, 1)
        output_pil = transforms.ToPILImage()(output_image[0].cpu())
        # crop to original size
        output_pil = output_pil.crop((
            pad_left,  # top left corner
            pad_top,   # top boundary
            pad_left + ori_width,  # right boundary
            pad_top + ori_height   # bottom boundary
        ))
        if args.resize:
            output_pil = img_shrink(output_pil)
        os.makedirs(args.output_dir, exist_ok=True)
        output_pil.save(os.path.join(args.output_dir, bname))
        print(f"Saved enhanced image to {os.path.join(args.output_dir, bname)}")
    # Calculate the average inference time, excluding the first few for stabilization
    if len(time_records) > 3:
        average_time = np.mean(time_records[3:])
    else:
        average_time = np.mean(time_records)
    print(f"Average inference time: {average_time:.4f} seconds")

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', '-i', type=str, default=None, help="path to the input image")
    parser.add_argument('--output_dir', '-o', type=str, default=None, help="the directory to save the output")
    parser.add_argument("--pretrained_model_path", type=str, default='stabilityai/stable-diffusion-2-1-base')
    parser.add_argument('--pretrained_path', type=str, default=None, help="path to a model state dict to be used")
    parser.add_argument('--seed', type=int, default=42, help="Random seed to be used")
    parser.add_argument("--align_method", type=str, choices=['wavelet', 'adain', 'nofix'], default="adain")
    parser.add_argument("--lambda_pix", default=1.0, type=float, help="the scale for pixel-level enhancement")
    parser.add_argument("--lambda_sem", default=1.0, type=float, help="the scale for sementic-level enhancements")
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=224)
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024)
    parser.add_argument("--latent_tiled_size", type=int, default=96) 
    parser.add_argument("--latent_tiled_overlap", type=int, default=32) 
    parser.add_argument("--mixed_precision", type=str, default="fp16")
    parser.add_argument("--default",  action="store_true", help="use default or adjustale setting?") 

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    args = set_args()
    args.pretrained_path = f'checkpoints/model.pkl'
    args.input_image = 'example/InputImg'
    args.output_dir = 'example/OutputImg'
    os.makedirs(args.output_dir, exist_ok=True)

    args.resize = False
    pisa_sr(args)