import os
import time
import subprocess
from glob import glob
import pandas as pd
import torch ## added

models = {
    'ckpt/unixgen_lightning.ckpt'
    : [
        ['fixed_each_unified', 1, 1],

        ['fixed_each_unified', 1, 2],
        ['fixed_each_unified', 2, 2],

        ['fixed_each_unified', 1, 3],
        ['fixed_each_unified', 2, 3],
        ['fixed_each_unified', 3, 3],
    ]
}

test_meta_files = ['metadata/mimiccxr_test_filtered.csv'] ## Specify filtered test metadata

round_idx = 0 ## added for tracking
total_rounds = sum(len(configs) for configs in models.values()) * len(test_meta_files)  ## added for tracking

for model_path, configs in models.items():
    for config in configs:
        for ckpt in [model_path]:
            for meta_file in test_meta_files:
                df = pd.read_csv(meta_file)

                if df.empty:
                    print(f"❌ Warning: {meta_file} is empty! Skipping this run.")
                    continue  # Skip to the next config

                round_idx += 1        ## added for tracking
                print(f"\n\n━━━━━━━━━━━━━━━ 🛰️ Starting Testing Round {round_idx} / {total_rounds} ━━━━━━━━━━━━━━━")## added for tracking
                print(f"📋 Model checkpoint: {ckpt}")                                                               ## added for tracking
                print(f"📋 Config: under_sample={config[0]}, max_img_num={config[1]}, target_count={config[2]}")    ## added for tracking
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

                TRAINING_CONFIG_LIST = list()
                for (k, v) in list(TRAINING_CONFIG.items()):
                    if (isinstance(v, bool)):
                        if v:
                            TRAINING_CONFIG_LIST.append("--{}={}".format(k, v))
                    else:
                        TRAINING_CONFIG_LIST.append("--{}={}".format(k, v))

                print('Training_lst:', TRAINING_CONFIG_LIST)
                subprocess.run(['python', SRC_PATH] + TRAINING_CONFIG_LIST)
                
                print(f"✅ Finished Testing Round {round_idx} / {total_rounds}")
                print(f"🧹 Clearing CUDA cache...\n")
                torch.cuda.empty_cache()
                
                time.sleep(10)
                print("━" * 10, "\n\n")
