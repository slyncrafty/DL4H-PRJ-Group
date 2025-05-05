import os
import argparse
import glob
import pandas as pd
import nltk
from bert_score import score as bertscore

nltk.download('punkt')

def evaluate_bleu(references, hypotheses):
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    smooth = SmoothingFunction().method1
    bleu_scores = [
        sentence_bleu([ref.split()], hyp.split(), smoothing_function=smooth)
        for ref, hyp in zip(references, hypotheses)
    ]
    avg_bleu = sum(bleu_scores)/len(bleu_scores)
    print(f"✅ Average BLEU: {avg_bleu:.4f}")
    return avg_bleu, bleu_scores

def evaluate_bertscore(references, hypotheses):
    P, R, F1 = bertscore(hypotheses, references, lang='en')
    avg_f1 = F1.mean().item()
    print(f"✅ Average BERTScore F1: {avg_f1:.4f}")
    return avg_f1, F1.tolist()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--decoded_glob', type=str, required=True,
                        help='Glob for your decoded CSVs e.g. "./output/*_GT_vs_GEN.csv"')
    parser.add_argument('--output_csv',  type=str, required=True,
                        help='Where to save the final evaluation CSV')
    parser.add_argument('--use_chexbert', action='store_true',
                        help='Also run CheXbert (requires chexbert-model installed)')
    parser.add_argument('--chexbert_model_path', type=str, default='chexbert-labeler.pth')
    args = parser.parse_args()

    # 1) load only CSVs
    print("🔍 Loading decoded CSVs…")
    dfs = []
    file_metrics = []
    for path in glob.glob(args.decoded_glob):
        if not path.lower().endswith('.csv'):
            print(f"⚠️  skipping non‑CSV: {path}")
            continue
        print("  •", path)
        df = pd.read_csv(path)
        # expect columns named exactly GroundTruth, Generated
        if 'GroundTruth' not in df.columns or 'Generated' not in df.columns:
            raise ValueError(f"{path} missing GroundTruth/Generated columns")
        # rename to unified names
        df = df.rename(columns={'GroundTruth':'GT_text','Generated':'gen_text'})
        dfs.append(df[['GT_text','gen_text']])
        # compute this file’s avg BLEU & avg BERTScore
        refs = df['GT_text'].astype(str).tolist()
        hyps = df['gen_text'].astype(str).tolist()
        _, b_scores    = evaluate_bleu(refs, hyps)
        _, bert_scores = evaluate_bertscore(refs, hyps)
        file_metrics.append({
            'file':           os.path.basename(path),
            'avg_BLEU':       sum(b_scores)/len(b_scores),
            'avg_BERTScore':  sum(bert_scores)/len(bert_scores)
        })
 
    full = pd.concat(dfs, ignore_index=True)
    print(f"📊 Total decoded samples: {len(full)}")

    # 2) metrics
    references = full['GT_text'].astype(str).tolist()
    hypotheses = full['gen_text'].astype(str).tolist()

    _, bleu_scores    = evaluate_bleu(references, hypotheses)
    _, bert_f1_scores = evaluate_bertscore(references, hypotheses)

    # 3) build & save
    full['BLEU']         = bleu_scores
    full['BERTScore_F1'] = bert_f1_scores
    full.to_csv(args.output_csv, index=False)
    print("✅ Saved full evaluation to", args.output_csv)
    
    # 4) per‑file summary
    per_file_df = pd.DataFrame(file_metrics)
    per_file_csv = args.output_csv.replace('.csv','') + '_per_file_summary.csv'
    per_file_df.to_csv(per_file_csv, index=False)
    print("✅ Saved per‑file summary to", per_file_csv)

    # 5) top‑3 slices
    p = args.output_csv.replace('.csv','') + '_top3_BLEU.csv'
    full.nlargest(3,'BLEU')[['GT_text','gen_text','BLEU']].to_csv(p, index=False)
    print("✅ Saved top‑3 BLEU to", p)

    q = args.output_csv.replace('.csv','') + '_top3_BERTScore.csv'
    full.nlargest(3,'BERTScore_F1')[['GT_text','gen_text','BERTScore_F1']].to_csv(q, index=False)
    print("✅ Saved top‑3 BERTScore to", q)

    # optional CheXbert
    if args.use_chexbert:
        from chexbert import CheXbertLabeler
        from sklearn.metrics import f1_score
        import numpy as np

        print("🔍 Running CheXbert labeler…")
        labeler = CheXbertLabeler(args.chexbert_model_path)
        gt_labels   = labeler.get_labels(references)
        pred_labels = labeler.get_labels(hypotheses)

        per_label_f1 = []
        for i in range(gt_labels.shape[1]):
            per_label_f1.append(f1_score(gt_labels[:,i], pred_labels[:,i]))
        chex_df = pd.DataFrame({
            'label': labeler.label_names,
            'F1': per_label_f1
        })
        chex_df.to_csv('chexbert_per_label_f1.csv', index=False)
        print("✅ Saved CheXbert per‑label F1 to chexbert_per_label_f1.csv")

if __name__=='__main__':
    main()
