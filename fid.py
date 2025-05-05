import os
import random
import numpy as np
from glob import glob
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import csv

import torch
import torchxrayvision as xrv
import skimage.io
import torchvision.transforms as TF
from torch.nn.functional import adaptive_avg_pool2d
from tqdm import tqdm
from scipy import linalg

# ─── Helpers ───────────────────────────────────────────────────────────────

IMAGE_EXTS = ("png","jpg","jpeg","bmp","tif","tiff","webp")

def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files):
        self.files = files
    def __len__(self):
        return len(self.files)
    def __getitem__(self, i):
        path = self.files[i]
        img = skimage.io.imread(path)
        img = xrv.datasets.normalize(img, 255)
        if img.ndim == 3:
            img = img.mean(2)[None,...]
        else:
            img = img[None,...]
        # center‐crop + resize to 224
        return TF.Compose([
            xrv.datasets.XRayCenterCrop(),
            xrv.datasets.XRayResizer(224),
            lambda x: torch.from_numpy(x)
        ])(img)

def get_activations(files, model, bs, dims, device, nw):
    ds = ImagePathDataset(files)
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, num_workers=nw, shuffle=False)
    feats = np.empty((len(files), dims), dtype=np.float32)
    idx = 0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dl, desc="Extracting features"):
            batch = batch.to(device)
            out = model.features(batch)
            if out.size(2)!=1 or out.size(3)!=1:
                out = adaptive_avg_pool2d(out, (1,1))
            out = out.squeeze(-1).squeeze(-1).cpu().numpy()
            feats[idx:idx+out.shape[0]] = out
            idx += out.shape[0]
    return feats

def calc_stats(files, model, bs, dims, device, nw):
    act = get_activations(files, model, bs, dims, device, nw)
    return act.mean(axis=0), np.cov(act, rowvar=False)

def frechet(mu1, s1, mu2, s2, eps=1e-6):
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(s1.dot(s2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(s1.shape[0]) * eps
        covmean = linalg.sqrtm((s1+offset).dot(s2+offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2*np.trace(covmean)

def calc_fid(gt_files, gen_files, bs, device, dims, nw):
    model = xrv.models.DenseNet(weights="densenet121-res224-all").to(device)
    mu1, s1 = calc_stats(gt_files, model, bs, dims, device, nw)
    mu2, s2 = calc_stats(gen_files, model, bs, dims, device, nw)
    return frechet(mu1, s1, mu2, s2)

# ─── Process one folder ─────────────────────────────────────────────────────

def process_folder(path, args, device):
    # collect all images in this folder
    files = []
    for ext in IMAGE_EXTS:
        files += glob(os.path.join(path, f"*.{ext}"))
    files = sorted(files)
    gt = [f for f in files if "GT"  in os.path.basename(f)]
    gn = [f for f in files if "gen" in os.path.basename(f)]
    print(f"→ {path} : {len(gt)} GT, {len(gn)} gen")
    if not gt or not gn or len(gt)!=len(gn):
        print("   ✗ skipping (empty or mismatch)")
        return None
    fid_value = calc_fid(gt, gn, args.batch_size, device, args.dims, args.num_workers)
    print(f"   ✓ FID = {fid_value:.3f}")
    return round(fid_value,3)

# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    p = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument('--gt_path',     type=str, required=True,
                   help='Top folder containing subfolders (or images) to score')
    p.add_argument('--dims',        type=int, default=1024)
    p.add_argument('--batch-size',  type=int, default=50)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--device',      type=str, default='cuda')
    args = p.parse_args()

    set_seed(123)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # determine folders to scan:
    to_process = []
    # 1) flat images in gt_path?
    found = False
    for ext in IMAGE_EXTS:
        if glob(os.path.join(args.gt_path, f"*.{ext}")):
            found = True
            break
    if found:
        to_process.append(args.gt_path)
    # 2) any subdirs with images?
    for sub in sorted(glob(os.path.join(args.gt_path, "*"))):
        if os.path.isdir(sub):
            for ext in IMAGE_EXTS:
                if glob(os.path.join(sub, f"*.{ext}")):
                    to_process.append(sub)
                    break

    # run
    results = []
    for folder in to_process:
        fid = process_folder(folder, args, device)
        if fid is not None:
            results.append({'folder':os.path.basename(folder), 'fid_score':fid})

    # write summary CSV
    out_csv = os.path.join(args.gt_path, "fid_scores_summary.csv")
    with open(out_csv,'w',newline='') as f:
        w = csv.DictWriter(f, fieldnames=['folder','fid_score'])
        w.writeheader()
        for row in results:
            w.writerow(row)

    print(f"\nDone! Wrote {len(results)} rows to {out_csv}")

if __name__=="__main__":
    main()
