import argparse
import os
import glob
import cv2
import numpy as np
import torch
import sys

# Ensure the repo root is on the PYTHONPATH
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

from basicsr.archs.craft_arch import CRAFT


def main():
    parser = argparse.ArgumentParser(description='CRAFT SISR Inference (no GT, no metrics)')
    parser.add_argument('--input',      type=str, required=True,
                        help='Folder containing low-resolution input images')
    parser.add_argument('--output',     type=str, required=True,
                        help='Folder to save super-resolved images')
    parser.add_argument('--scale',      type=int, default=4,
                        help='Super-resolution scale factor')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained CRAFT .pth model')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = CRAFT(
        upscale=args.scale,
        in_chans=3,
        img_size=64,
        window_size=16,
        img_range=1.,
        depths=[2, 2, 2, 2],
        embed_dim=48,
        num_heads=[6, 6, 6, 6],
        mlp_ratio=2,
        resi_connection='1conv'
    ).to(device)
    model.eval()

    ckpt = torch.load(args.model_path, map_location=device)
    key = 'params_ema' if 'params_ema' in ckpt else 'params'
    model.load_state_dict(ckpt[key], strict=True)

    window_size = 16

    for img_path in sorted(glob.glob(os.path.join(args.input, '*'))):
        name, ext = os.path.splitext(os.path.basename(img_path))

        # Read and normalize
        img = cv2.imread(img_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
        h, w = img.shape[:2]

        # BGR->RGB, HWC->CHW, ensure contiguous strides
        arr = img[:, :, ::-1].transpose(2, 0, 1).copy()
        inp = torch.from_numpy(arr).unsqueeze(0).to(device)

        with torch.no_grad():
            _, _, h_old, w_old = inp.size()
            # pad to window_size multiple
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            inp_pad = torch.cat([inp, torch.flip(inp, [2])], 2)[:, :, :h_old + h_pad, :]
            inp_pad = torch.cat([inp_pad, torch.flip(inp_pad, [3])], 3)[:, :, :, :w_old + w_pad]

            out = model(inp_pad)
            out = out[..., :h_old * args.scale, :w_old * args.scale]

        # to CPU and uint8
        out_img = out.squeeze().cpu().clamp(0, 1).numpy()
        out_img = (out_img.transpose(1, 2, 0)[..., ::-1] * 255.0).round().astype(np.uint8)

        out_file = os.path.join(args.output, f"{name}_CRAFT.png")
        cv2.imwrite(out_file, out_img)
        print(f"Saved: {out_file}")


if __name__ == '__main__':
    main()
