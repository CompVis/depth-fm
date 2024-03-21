import os
import torch
import einops
import argparse
import numpy as np
from PIL import Image
from depthfm import DepthFM
import matplotlib.pyplot as plt


def load_im(fp):
    assert os.path.exists(fp), f"File not found: {fp}"
    im = Image.open(fp).convert('RGB')
    x = np.array(im)
    x = einops.rearrange(x, 'h w c -> c h w')
    x = x / 127.5 - 1
    x = torch.tensor(x, dtype=torch.float32)[None]
    return x


def main(args):
    print(f"{'Input':<10}: {args.img}")
    print(f"{'Steps':<10}: {args.num_steps}")
    print(f"{'Ensemble':<10}: {args.ensemble_size}")

    # Load the model
    model = DepthFM(args.ckpt)
    model.cuda().eval()

    # Load an image
    im = load_im(args.img).cuda()

    # Generate depth
    depth = model.predict_depth(im, num_steps=args.num_steps, ensemble_size=args.ensemble_size)
    depth = depth.squeeze(0).squeeze(0).cpu().numpy()       # (h, w) in [0, 1]

    # Convert depth to [0, 255] range
    if args.no_color:
        depth = (depth * 255).astype(np.uint8)
    else:
        depth = plt.get_cmap('magma')(depth, bytes=True)[..., :3]

    # Save the depth map
    depth_fp = args.img.replace('.png', '-depth.png')
    Image.fromarray(depth).save(depth_fp)
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
    args = parser.parse_args()

    main(args)
