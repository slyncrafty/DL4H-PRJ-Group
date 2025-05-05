# unified_run_ablation.py
import os
import time
import subprocess
from glob import glob
import pandas as pd
import torch 

single_view_configs = [
    ['fixed_each_unified', 1, 1],
    ['fixed_each_unified', 1, 2],
]
multi_view_configs = [
    ['fixed_each_unified', 2, 3],
    ['fixed_each_unified', 3, 3],
]

test_meta_files = [
    ('metadata/single_view.csv', single_view_configs),
    ('metadata/multi_view.csv',  multi_view_configs),
] ## Specify filtered test metadata

round_idx = 0 ## added for tracking
total_rounds = sum(len(cfgs) for _, cfgs in test_meta_files)

for meta_file, configs in test_meta_files:
    for config in configs:
        ckpt = 'ckpt/unixgen_lightning.ckpt'
        df = pd.read_csv(meta_file)
        if df.empty:
            print(f"❌ Warning: {meta_file} is empty! Skipping this run.")
            continue

        round_idx += 1
        print(f"\n\n━━━━━━━━━━━━━━━ 🛰️ Starting Testing Round {round_idx} / {total_rounds} ━━━━━━━━━━━━━━━")
        print(f"📋 Model checkpoint: {ckpt}")
        print(f"📋 Config: under_sample={config[0]}, max_img_num={config[1]}, target_count={config[2]}")
        print(f"📋 Test metadata file: {meta_file}")

        EXP_PATH = os.getcwd()
        SRC_PATH = 'unified_main.py'
        TRAINING_CONFIG = {
            'test': True,
            'reload_ckpt_dir': ckpt,
            'under_sample': config[0],
            'max_img_num': config[1],
            'target_count': config[2],
            'test_meta_file': meta_file,
        }

        TRAINING_CONFIG_LIST = [f"--{k}={v}" for k, v in TRAINING_CONFIG.items() if not isinstance(v, bool) or v]

        print('Training_lst:', TRAINING_CONFIG_LIST)
        subprocess.run(['python', SRC_PATH] + TRAINING_CONFIG_LIST)

        print(f"✅ Finished Testing Round {round_idx} / {total_rounds}")
        print(f"🧹 Clearing CUDA cache...\n")
        torch.cuda.empty_cache()
        time.sleep(10)
        print("━" * 10, "\n\n")
