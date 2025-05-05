# decode_report.py
import os
import glob
import argparse
import torch
import pandas as pd
from nltk.translate.bleu_score import corpus_bleu
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

def decode_report(
    test_output_dir,
    tokenizer_dir='BBPE_tokenizer',
    save_csv=True,
    save_dir='output'
):
    # — validate inputs —
    assert os.path.isdir(test_output_dir), f"No such directory: {test_output_dir}"

    # — load tokenizer —
    tokenizer = ByteLevelBPETokenizer(
        os.path.join(tokenizer_dir, 'vocab.json'),
        os.path.join(tokenizer_dir, 'merges.txt')
    )
    tokenizer.add_special_tokens(["[PAD]", "[SOS]", "[EOS]", "[SEP]", "[MASK]"])
    tokenizer._tokenizer.post_processor = BertProcessing(
        ("[EOS]", tokenizer.token_to_id("[EOS]")),
        ("[SOS]", tokenizer.token_to_id("[SOS]")),
    )
    tokenizer.enable_truncation(max_length=256)
    tokenizer.enable_padding(
        pad_id=tokenizer.token_to_id("[PAD]"),
        pad_token="[PAD]",
        length=256
    )

    # — find all .pt outputs in the given folder —
    pt_files = sorted(glob.glob(os.path.join(test_output_dir, "test_output_*.pt")))
    assert pt_files, f"No test_output_*.pt found in {test_output_dir}"

    # we will collect a summary of BLEU for each file
    summary_records = []

    # process each checkpoint file
    for pt_path in pt_files:
        print(f"📦 Loading: {pt_path}")
        outputs = torch.load(pt_path, map_location='cpu')

        GT_texts, GEN_texts = [], []

        # flatten all batches
        for batch_out in outputs:
            gt = batch_out['GT_text']    # tensor [B, T]
            gen = batch_out['gen_text']  # tensor [B, T]
            for gti, geni in zip(gt, gen):
                # decode each sequence
                gt_dec = tokenizer.decode(gti.tolist(), skip_special_tokens=True)
                gen_dec = tokenizer.decode(geni.tolist(), skip_special_tokens=True)
                GT_texts.append(gt_dec)
                GEN_texts.append(gen_dec)

        # save side‑by‑side CSV
        if save_csv:
            os.makedirs(save_dir, exist_ok=True)
            base = os.path.basename(pt_path).replace('.pt','')
            df = pd.DataFrame({
                "GroundTruth": GT_texts,
                "Generated": GEN_texts
            })
            csv_path = os.path.join(save_dir, f"{base}_GT_vs_GEN.csv")
            df.to_csv(csv_path, index=False)
            print(f"✅ Wrote examples to {csv_path}")

        # compute corpus BLEU
        refs = [[r.split()] for r in GT_texts]
        cands = [h.split() for h in GEN_texts]
        bleu1 = corpus_bleu(refs, cands, weights=(1,0,0,0))
        bleu2 = corpus_bleu(refs, cands, weights=(0.5,0.5,0,0))
        bleu3 = corpus_bleu(refs, cands, weights=(1/3,1/3,1/3,0))
        bleu4 = corpus_bleu(refs, cands, weights=(0.25,0.25,0.25,0.25))

        print(f"🧾 {pt_path}: BLEU‑1={bleu1:.4f}, BLEU‑2={bleu2:.4f}, BLEU‑3={bleu3:.4f}, BLEU‑4={bleu4:.4f}")

        summary_records.append({
            'file': os.path.basename(pt_path),
            'samples': len(GT_texts),
            'BLEU-1': round(bleu1,4),
            'BLEU-2': round(bleu2,4),
            'BLEU-3': round(bleu3,4),
            'BLEU-4': round(bleu4,4)
        })

    # write summary CSV
    summary_df = pd.DataFrame(summary_records)
    summary_csv = os.path.join(save_dir, "bleu_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"📊 Saved BLEU summary to {summary_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--test_output_dir',
        type=str,
        required=True,
        help='Directory containing test_output_*.pt files'
    )
    parser.add_argument('--tokenizer_dir', type=str, default='BBPE_tokenizer')
    parser.add_argument('--save_csv', action='store_true', help='Save per‑sample CSVs')
    parser.add_argument('--save_dir', type=str, default='output')
    args = parser.parse_args()

    decode_report(
        test_output_dir=args.test_output_dir,
        tokenizer_dir=args.tokenizer_dir,
        save_csv=args.save_csv,
        save_dir=args.save_dir
    )
