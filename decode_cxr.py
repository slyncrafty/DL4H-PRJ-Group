import os
import argparse
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import numpy as np

from vae import VQGanVAE
from helpers import str2bool


def decode_single_output(output, vae, save_dir, img_save=True, preview=False):
    print(f"🧠 Decoding {len(output)} batches...")
    preview_done = False

    for i, row in tqdm(enumerate(output), desc='Batches', total=len(output)):
        max_img_num = sum(k.startswith('GT_image') for k in row.keys())
        bsz = len(row['img_paths'])

        for b in range(bsz):
            name_paths = row['img_paths'][b].split('|')[0].split('/')
            name = name_paths[-4] + "_" + name_paths[-3] + "_" + name_paths[-2]

            if preview and not preview_done:
                fig, axs = plt.subplots(2, max_img_num, figsize=(4 * max_img_num, 6))
                axs = np.array([[axs[0]]] * 2) if max_img_num == 1 else np.atleast_2d(axs)

            for img_idx in range(1, max_img_num + 1):
                GT_tensor = row[f'GT_image{img_idx}'].reshape(-1, row[f'GT_image{img_idx}'].shape[-1])[b][1:-1].unsqueeze(0)
                gen_tensor = row[f'gen_image{img_idx}'].reshape(-1, row[f'gen_image{img_idx}'].shape[-1])[b][1:-1].unsqueeze(0)

                GT_img = vae.decode(GT_tensor)[0].permute(1, 2, 0).detach().cpu().numpy()
                gen_img = vae.decode(gen_tensor)[0].permute(1, 2, 0).detach().cpu().numpy()
                torch.cuda.empty_cache()

                if img_save:
                    plt.imsave(os.path.join(save_dir, name + f'_gen_img{img_idx}.jpeg'), gen_img)
                    plt.imsave(os.path.join(save_dir, name + f'_GT_img{img_idx}.jpeg'), GT_img)

                if preview and not preview_done and b < 3:
                    axs[0, img_idx - 1].imshow(GT_img)
                    axs[0, img_idx - 1].set_title(f"GT Image {img_idx}")
                    axs[0, img_idx - 1].axis('off')
                    axs[1, img_idx - 1].imshow(gen_img)
                    axs[1, img_idx - 1].set_title(f"Gen Image {img_idx}")
                    axs[1, img_idx - 1].axis('off')

            if preview and not preview_done and b == 2:
                plt.tight_layout()
                plt.show()
                preview_done = True


def decode_cxr_outputs(glob_pattern, save_dir, vqgan_model_path, vqgan_config_path, img_save=True, preview=True):
    pt_files = sorted(glob(glob_pattern))
    assert pt_files, f"No .pt files found for pattern: {glob_pattern}"

    print(f"🔓 Loading VQGAN model from {vqgan_model_path}")
    vae = VQGanVAE(vqgan_model_path, vqgan_config_path).cuda()

    for i, pt_file in enumerate(pt_files):
        print(f"\n📦 [{i+1}/{len(pt_files)}] Loading test output from {pt_file}")
        output = torch.load(pt_file, map_location='cuda')

        file_name = os.path.splitext(os.path.basename(pt_file))[0]
        output_save_dir = os.path.join(save_dir, file_name)
        os.makedirs(output_save_dir, exist_ok=True)

        decode_single_output(
            output, vae, output_save_dir,
            img_save=img_save,
            preview=(preview and i == 0)  # Only preview from first file
        )

    if img_save:
        print("\n📄 All images saved!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_output_glob', type=str, required=True,
                        help="Glob pattern for .pt files (e.g. ./output/test_output_*.pt)")
    parser.add_argument('--save_dir', type=str, default='./output/decoded_images/')
    parser.add_argument('--vqgan_model_path', type=str, default='mimiccxr_vqgan/last.ckpt')
    parser.add_argument('--vqgan_config_path', type=str, default='mimiccxr_vqgan/2021-12-17T08-58-54-project.yaml')
    parser.add_argument('--img_save', type=str2bool, default=True)
    parser.add_argument('--preview', type=str2bool, default=True)
    args = parser.parse_args()

    decode_cxr_outputs(
        glob_pattern=args.test_output_glob,
        save_dir=args.save_dir,
        vqgan_model_path=args.vqgan_model_path,
        vqgan_config_path=args.vqgan_config_path,
        img_save=args.img_save,
        preview=args.preview
    )
