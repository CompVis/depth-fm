import os
import torch
import einops
import argparse
import numpy as np
from PIL import Image
from PIL.Image import Resampling
from depthfm import DepthFM
import matplotlib.pyplot as plt

def get_dtype_from_str(dtype_str):
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype_str]

def resize_max_res(
    img: Image.Image, max_edge_resolution: int, resample_method=Resampling.BILINEAR
) -> Image.Image:
    """
    Resize image to limit maximum edge length while keeping aspect ratio.

    Args:
        img (`Image.Image`):
            Image to be resized.
        max_edge_resolution (`int`):
            Maximum edge length (pixel).
        resample_method (`PIL.Image.Resampling`):
            Resampling method used to resize images.

    Returns:
        `Image.Image`: Resized image.
    """
    original_width, original_height = img.size
    downscale_factor = min( max_edge_resolution / original_width, max_edge_resolution / original_height)

    new_width  = int(original_width * downscale_factor)
    new_height = int(original_height * downscale_factor)

    new_width  = round(new_width / 64) * 64
    new_height = round(new_height / 64) * 64

    print(f"Resizing image from {original_width}x{original_height} to {new_width}x{new_height}")

    resized_img = img.resize((new_width, new_height), resample=resample_method)
    return resized_img, (original_width, original_height)

def load_im(fp, processing_res=-1):
    assert os.path.exists(fp), f"File not found: {fp}"
    im = Image.open(fp).convert('RGB')
    if processing_res < 0:
        processing_res = max(im.size)
    im, orig_res = resize_max_res(im, processing_res)
    x = np.array(im)
    x = einops.rearrange(x, 'h w c -> c h w')
    x = x / 127.5 - 1
    x = torch.tensor(x, dtype=torch.float32)[None]
    return x, orig_res


def main(args):
    print(f"{'Input':<10}: {args.img}")
    print(f"{'Steps':<10}: {args.num_steps}")
    print(f"{'Ensemble':<10}: {args.ensemble_size}")

    # Load the model
    model = DepthFM(args.ckpt)
    model.cuda(args.device).eval()

    # Load an image
    im, orig_res = load_im(args.img, args.processing_res)
    im = im.cuda(args.device)

    # Generate depth
    dtype = get_dtype_from_str(args.dtype)
    model.model.dtype = dtype
    with torch.autocast(device_type="cuda", dtype=dtype):
        depth = model.predict_depth(im, num_steps=args.num_steps, ensemble_size=args.ensemble_size)
    depth = depth.squeeze(0).squeeze(0).cpu().numpy()       # (h, w) in [0, 1]

    # Convert depth to [0, 255] range
    if args.no_color:
        depth = (depth * 255).astype(np.uint8)
    else:
        depth = plt.get_cmap('magma')(depth, bytes=True)[..., :3]

    # Save the depth map
    depth_fp = args.img + '_depth.png'
    depth_img = Image.fromarray(depth)
    if depth_img.size != orig_res:
        depth_img = depth_img.resize(orig_res, Resampling.BILINEAR)
    depth_img.save(depth_fp)
    print(f"==> Saved depth map to {depth_fp}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("DepthFM Inference")
    parser.add_argument("--img", type=str, default="assets/dog.png",
                        help="Path to the input image")
    parser.add_argument("--ckpt", type=str, default="checkpoints/depthfm-v1.ckpt",
                        help="Path to the model checkpoint")
    parser.add_argument("--num_steps", type=int, default=2,
                        help="Number of steps for ODE solver")
    parser.add_argument("--ensemble_size", type=int, default=4,
                        help="Number of ensemble members")
    parser.add_argument("--no_color", action="store_true",
                        help="If set, the depth map will be grayscale")
    parser.add_argument("--device", type=int, default=0,
                        help="GPU to use")
    parser.add_argument("--processing_res", type=int, default=-1, 
                        help="Longer edge of the image will be resized to this resolution. -1 to disable resizing.")
    parser.add_argument("--dtype", type=str, choices=["fp32", "bf16", "fp16"], default="fp16", 
                        help="Run with specific precision. Speeds up inference with subtle loss")
    args = parser.parse_args()

    main(args)
