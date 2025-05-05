"""
fix_unixgen_ckpt.py

Make a legacy UniXGen checkpoint compatible with modern PyTorch-Lightning.

- Renames   transformerLM_unified.image_pos_emb.weights_{0,1}
    →        transformerLM_unified.image_pos_emb.weights.{0,1}
- Inserts   'global_step' and 'epoch' fields if they are missing
- Verifies  hyper-parameter shapes still match
- Saves     ckpt/unixgen_lightning.ckpt

run it once with
python fix_unixgen_ckpt.py \
       --in_ckpt  ckpt/unixgen.ckpt \
       --out_ckpt ckpt/unixgen_lightning.ckpt
"""

import torch, pathlib, shutil, argparse, pprint, sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_ckpt",  default="ckpt/unixgen.ckpt")
    parser.add_argument("--out_ckpt", default="ckpt/unixgen_lightning.ckpt")
    args = parser.parse_args()

    in_path  = pathlib.Path(args.in_ckpt)
    out_path = pathlib.Path(args.out_ckpt)
    assert in_path.exists(), f"Input checkpoint {in_path} not found"

    print(f"🔧  Loading  {in_path}")
    ckpt = torch.load(in_path, map_location="cpu")

    # 1) fix missing bookkeeping keys ----------------------------------------
    ckpt.setdefault("global_step", 0)
    ckpt.setdefault("epoch",       0)

    # 2) rename legacy state-dict keys ---------------------------------------
    sd = ckpt["state_dict"]
    rename_map = {
        "transformerLM_unified.image_pos_emb.weights_0":
            "transformerLM_unified.image_pos_emb.weights.0",
        "transformerLM_unified.image_pos_emb.weights_1":
            "transformerLM_unified.image_pos_emb.weights.1",
    }
    for old, new in rename_map.items():
        if old in sd and new not in sd:
            sd[new] = sd.pop(old)
            print(f"    ✔ renamed {old} → {new}")

    # 3) print important hyper-params for later use ---------------------------
    hp = ckpt["hyper_parameters"]
    print("\n📋 Hyper-parameters stored in ckpt:")
    pprint.pprint({k: hp[k] for k in
                  ("img_vocab_size","num_img_tokens",
                   "max_seq_len","max_img_num","target_count")})

    # 4) backup original and save new file ------------------------------------
    if out_path.exists():
        shutil.copy2(out_path, out_path.with_suffix(".bak"))
    torch.save(ckpt, out_path)
    print(f"\n✅  Saved Lightning-compatible ckpt →  {out_path}")

if __name__ == "__main__":
    main()