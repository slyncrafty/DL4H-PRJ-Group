# UniXGen on Colab

📋 This repository re-implements **UniXGen** (the multimodal radiology report + image generator from MIMIC-CXR) to run end-to-end on Google Colab (single-GPU setup).

- 🧐 [Vision-Language Generative Model for
  View-Specific Chest X-ray Generation](https://arxiv.org/pdf/2302.12172)
- 📎 [UniXGen Original Git Repository](https://github.com/ttumyche/UniXGen)

  > **BibTeX**:

        @article{lee2023vision,
        title   = {Vision-Language Generative Model for View-Specific Chest X-ray Generation},
        author  = {Lee, Hyungyung and Lee, Da Young and Kim, Wonjae and Kim, Jin-Hwa and Kim, Tackeun and Kim, Jihang and Sunwoo, Leonard and Choi, Edward},
        journal = {arXiv preprint arXiv:2302.12172},
        volume  = {2023},
        year    = {2023},
        doi     = {10.48550/arXiv.2302.12172},
        url     = {https://doi.org/10.48550/arXiv.2302.12172},
        eprint  = {2302.12172},
        archivePrefix = {arXiv},
        primaryClass  = {eess.IV}
        }

It supports:

- **Training** from scratch or fine-tuning
- **Inference** / ablations over single vs. multi-view inputs
- **Evaluation** (BLEU, BERTScore, CheXbert, FID)
- **Automated scripts** for batching experiments

![Original UniXGen Model Architecture](https://user-images.githubusercontent.com/123858584/226160635-ff47d23a-e35f-45ec-aeb0-06a823f50e5d.png)
Figure 2: Overview of ViewXGen architecture from [Vision-Language Generative Model for
View-Specific Chest X-ray Generation](https://arxiv.org/pdf/2302.12172)

## Set up & Requirements

🔑 Example Setup: [UniXGenOnColab.ipynb](https://github.com/slyncrafty/DL4H-PRJ-Group/blob/main/UniXGenOnColab.ipynb)

🔑 Clone this repo:

```clone git
git clone https://github.com/slyncrafty/DL4H-PRJ-Group.git UniXGen
```

🔑 To install requirements in Google Colab:

```disk mount
from google.colab import drive
drive.mount('/content/drive')
```

```
PRJ_ROOT ='/content/drive/MyDrive/UniXGen'
DATA_PATH = '/content/drive/MyDrive/UniXGen/data'
%cd {PRJ_ROOT}
```

```setup
# install core deps
%cd {PRJ_ROOT}
%pip install --upgrade pip
%pip install -r requirements.txt
%pip install pytorch-lightning==2.0.9 --force-reinstall
%pip install --force-reinstall torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118
%pip uninstall -y numpy
%pip install numpy==1.24.4
%pip uninstall -y jax jaxlib
%pip install --upgrade jax==0.4.23 jaxlib==0.4.23
```

```external source
%cd {PRJ_ROOT}
!git clone https://github.com/CompVis/taming-transformers.git
%cd taming-transformers
!pip install -e .
%cd {PRJ_ROOT}
```

🔑 Replace file: {PRJ_ROOT}/taming-transformers/taming/data/utils.py with [utils.py](https://drive.google.com/file/d/1NCO8hojet42JdrgX1vKV3uMPpCLBWDw8/view?usp=drive_link)

🔑 Key versions (tested):

| Package / Library              | Version / Source                        |
| ------------------------------ | --------------------------------------- |
| **Python**                     | 3.11                                    |
| **CUDA**                       | 12.1 (for GPU acceleration)             |
| **torch**                      | `2.0.1+cu118`                           |
| **torchvision**                | `0.15.2+cu118`                          |
| **PyTorch Lightning**          | `2.0.9`                                 |
| **transformers**               | Latest tested (v4.37.2)                 |
| **tokenizers**                 | ByteLevelBPE                            |
| **taming‑transformers**        | custom clone from CompVis repo          |
| **lpips**                      | from `taming.modules.autoencoder.lpips` |
| **omegaconf**                  | `2.2.3`                                 |
| **pandas**                     | `2.0.3`                                 |
| **numpy**                      | `1.24.4`                                |
| **albumentations**             | `1.3.1`                                 |
| **axial-positional-embedding** | default(tested `0.3.12`)                |
| **matplotlib**                 | default                                 |
| **seaborn**                    | default                                 |
| **nltk**                       | default                                 |
| **tqdm**                       | default                                 |
| **scikit‑learn**               | `1.3.0`                                 |
| **torchxrayvision**            | `1.2.1`                                 |
| **jax**                        | `0.4.23 `                               |
| **jaxlib**                     | `0.4.23`                                |
| **bert‑score**                 | `0.3.13`                                |

## Setup Data

🔧Download MIMIC-CXR-JPG images & reports

- You must be a credential user defined in PhysioNet to access the data.
- Download chest X-rays from [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) and Place images under **data/images/**
- Download reports from [MIMIC-CXR Database](https://physionet.org/content/mimic-cxr/2.0.0/) and Place reports under **data/reports/**

🔧 Download VQGAN Tokenizer

- Download [Chest X-ray Tokenizer(VQGAN))](https://drive.google.com/drive/folders/1Ia_GqRrmZ8g6md02TC5_nkrGn6eUwVaG) and Place under **/mimiccxr_vqgan**

🔧 Place model file under **/ckpt**

- (Recommended) Download [Pre-updated UniXGen Model](https://drive.google.com/file/d/1LuZXq7DpQUV9cgWTLK6SRvlmSHu_a5E1/view?usp=drive_link) and Place model file under **/ckpt**

- (Optional) Download [UniXGen model](). This model requires update. See UniXGenOnColab.ipynb (Section Pre-processing - Fix ckpt version for compatibility(Optional))

## Metadata

🔧 **Original train, valid and test split sets and pre-filtered metadata CSVs provided under /metadata**

```
/metadata
          /mimiccxr_test_sub_final.csv
          /mimiccxr_train_sub_final.csv
          /mimiccxr_validate_sub_final.csv
          /mimiccxr_test_filtered.csv # smaller test split
          /multi_view.csv # for Ablation1
          /single_view.csv # for Ablation1
```

🔧 (Optional) Preprocess & filter only studies that actually exist on disk to generate your own `mimiccxr_test_filtered.csv`:
See UniXGenOnColab.ipynb(Section Pre-processing - Create Filtered metadata)

```
python scripts/preprocess_metadata.py \
 --original metadata/mimiccxr_test.csv \
 --image_dir data/images \
 --out_filtered metadata/mimiccxr_test_filtered.csv \
 --out_summary metadata/mimiccxr_test_summary.csv
```

🔧 (Optional) Generate ablation splits (single vs. multi-view):

```
python scripts/generate_experiment_csvs.py \
 --summary_csv metadata/mimiccxr_test_summary.csv \
 --filtered_csv metadata/mimiccxr_test_filtered.csv \
 --image_root data/images \
 --out_dir metadata
```

> This produces:
>
> - metadata/single_view.csv
> - metadata/multi_view.csv

## Training

To train the model(s) in the paper from scratch on MIMIC-CXR, run this command:

```train
python unified_main.py \
  --test=False \
  --train_meta_file metadata/mimiccxr_train.csv \
  --val_meta_file   metadata/mimiccxr_validate.csv \
  --lr 5e-4 \
  --batch_size 8 \
  --n_epochs 50 \
  --max_img_num 3 \
  --target_count 3 \
  --under_sample fixed_each_unified \
  --vqgan 512 \
  --vqgan_model_path mimiccxr_vqgan/last.ckpt \
  --vqgan_config_path mimiccxr_vqgan/config.yaml
```

> Original Authors' note on computation: Four NVIDIA RTX A6000 GPUs

## Testing

To run a test on the model, run this command:

```test
!python evaluate_outputs.py \
    --output_glob "./output/test_output_*.pt" \
    --output_csv "./output/summary_metrics_normal.csv"
```

> test output `.pt` files are saved under output/

## Decoding

1. Decode Images

```decode_cxr
!python decode_cxr.py \
  --test_output_glob="./output/test_output_*.pt" \
  --save_dir="./output/decoded_images/" \
  --vqgan_model_path="./mimiccxr_vqgan/last.ckpt" \
  --vqgan_config_path="./mimiccxr_vqgan/2021-12-17T08-58-54-project.yaml" \
  --img_save=True \
  --preview=True

```

> 📋 Decoded image files are saved under output/decoded_images

2. Decode Reports

```decode_reports
!python decode_report.py \
    --test_output_dir ./output \
    --tokenizer_dir BBPE_tokenizer \
    --save_csv \
    --save_dir ./output/decoded_reports

```

> 📋 Decoded report files and a BLEU score summary csv file are saved under output/decoded_reports

## Evaluation

To evaluate the outputs, run:

1. BLEU and BERT

```eval_
!python evaluate_outputs.py \
    --decoded_glob "./output/decoded_reports/*_GT_vs_GEN.csv" \
    --output_csv "./output/eval_summary.csv"
```

> 📋 evaluation summary files(.csv) are saved under output/.

2. FID

```eval_fid
!python fid.py --gt_path ./output/decoded_images \
              --batch-size 32 --dims 1024 --num-workers 8
```

> 📋 fid_scores_summary.csv file is saved under output/decoded_images.

## Pre-trained Models

You can download pretrained models here:

- [Chest X-ray Tokenizer(VQGAN))](https://drive.google.com/drive/folders/1Ia_GqRrmZ8g6md02TC5_nkrGn6eUwVaG?usp=sharing) and place into `/mimiccxr_vqgan`

- [Pre-updated UniXGen Model](https://drive.google.com/file/d/1LuZXq7DpQUV9cgWTLK6SRvlmSHu_a5E1/view?usp=drive_link) and place into `/ckpt`
  **OR** [UniXGen model](https://drive.google.com/file/d/1RBQEoYTBRKBh6L53QCE0OIXL0Da5knkY/view?usp=sharing) and place into `/ckpt`

## Results

Our model achieves the following performance on :

### Evaluation metrics results

| Group | Input  | BLEU‑1 | BLEU‑2 | BLEU‑3 | BLEU‑4 | BERT_F1 |  FID   |
| :---: | :----- | :----: | :----: | :----: | :----: | :-----: | :----: |
| S w/1 | 1 of 1 | 0.1870 | 0.0651 | 0.0230 | 0.0090 | 0.8150  | 40.784 |
| S w/2 | 1 of 2 | 0.2108 | 0.0738 | 0.0258 | 0.0100 | 0.8140  | 41.822 |
|       | 2 of 2 | 0.2147 | 0.0773 | 0.0267 | 0.0090 | 0.8140  | 39.838 |
| S w/3 | 1 of 3 | 0.2227 | 0.0823 | 0.0297 | 0.0000 | 0.8160  | 49.520 |
|       | 2 of 3 | 0.2288 | 0.0770 | 0.0265 | 0.0000 | 0.8140  | 41.276 |
|       | 3 of 3 | 0.1750 | 0.0644 | 0.0197 | 0.0140 | 0.8140  | 40.486 |

### Colab vs. Original Paper (BLEU‑4 & FID)

| Group | Input  | BLEU‑4 (Colab) | FID (Colab) | BLEU‑4 (Paper) | FID (Paper) |
| :---: | :----- | :------------: | :---------: | :------------: | :---------: |
| S w/1 | 1 of 1 |     0.0090     |   40.784    |     0.042      |    25.86    |
| S w/2 | 1 of 2 |     0.0100     |   41.822    |     0.056      |   16.965    |
|       | 2 of 2 |     0.0090     |   39.838    |     0.056      |    9.186    |
| S w/3 | 1 of 3 |     0.0000     |   49.520    |     0.054      |   21.148    |
|       | 2 of 3 |     0.0000     |   41.276    |     0.060      |   12.792    |
|       | 3 of 3 |     0.0140     |   40.486    |     0.063      |   12.684    |

### Ablation Results

| Group | Input  | **Unique‑view** BLEU‑4 | BERT F1 |  FID   | **Fixed‑order** BLEU‑4 | BERT F1 |  FID   |
| :---: | :----- | :--------------------: | :-----: | :----: | :--------------------: | :-----: | :----: |
| S w/1 | 1 of 1 |         0.0090         | 0.8150  | 40.784 |         0.0080         | 0.8160  | 41.907 |
| S w/2 | 1 of 2 |         0.0100         | 0.8140  | 41.822 |         0.0101         | 0.8140  | 41.656 |
|       | 2 of 2 |         0.0090         | 0.8140  | 39.838 |         0.0060         | 0.8170  | 53.043 |
| S w/3 | 1 of 3 |         0.0000         | 0.8160  | 49.520 |         0.0120         | 0.8130  | 38.343 |
|       | 2 of 3 |         0.0000         | 0.8140  | 41.276 |         0.0090         | 0.8150  | 39.682 |
|       | 3 of 3 |         0.0140         | 0.8140  | 40.486 |         0.0090         | 0.8150  | 39.643 |

## Colab Notebook

> [UniXGenOnColab.ipynb](https://github.com/slyncrafty/DL4H-PRJ-Group/blob/main/UniXGenOnColab.ipynb)
