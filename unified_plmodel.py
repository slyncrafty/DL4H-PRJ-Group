import os
import csv
import math
import time
import random
import numpy as np

import torch
from torch.optim import AdamW
from torch.nn.functional import cross_entropy

import pytorch_lightning as pl

from nltk.translate.bleu_score import corpus_bleu
from transformer_pytorch.transformer_unified import TransformerLM_unified
from transformers.optimization import get_cosine_schedule_with_warmup

random.seed(42)

class TransformerLightning_unified(pl.LightningModule):
    def __init__(self, lr=5e-4, weight_decay=0.01,
                 pad_token_idx=0, sos_token_idx=1, eos_token_idx=2,
                 save_dir="", causal_trans='conditioned_causal', **kargs):
        super().__init__()
        self.kargs = kargs
        self.max_img_num = kargs['max_img_num']
        self.under_sample = kargs['under_sample']
        self.attn_type = kargs['attn_type']
        self.num_txt_tokens = kargs['num_tokens']
        self.num_img_tokens = kargs['num_img_tokens']

        self.ckpt_path = kargs.get('reload_ckpt_dir', None)
        self.target_count = kargs.get('target_count', None)
        self.test_meta_file_name = kargs.get('test_meta_file', None)

        self.transformerLM_unified = TransformerLM_unified(**kargs)
        self.weight_decay = weight_decay
        self.lr = lr
        self.pad_token_idx = pad_token_idx
        self.sos_token_idx = sos_token_idx
        self.eos_token_idx = eos_token_idx
        self.save_dir = save_dir
        self.causal = causal_trans

        self.save_hyperparameters(ignore=['tokenizer'])

        self.test_outputs = []    ## Added

    def forward(self, batch):
        logit = self.transformerLM_unified(batch, causal=self.causal)
        return logit

    def training_step(self, batch, batch_idx):
        img1, txt, modes, view = batch['img1'], batch['txt'], batch['modes'], batch['view_position']

        assert txt.shape[0] == img1.shape[0]
        batch_size = txt.shape[0]
        txt_seq_len = txt.shape[1]
        img_seq_len = img1.shape[1]
        n = img_seq_len + txt_seq_len
        if 'img2' in batch.keys():
            img2 = batch['img2']
            n += img2.shape[1]
        if 'img3' in batch.keys():
            img3 = batch['img3']
            n += img3.shape[1]

        logit = self(batch)[:, :-1, :]
        max_neg_value = -torch.finfo(logit.dtype).max

        for bsz in range(batch_size):
            if np.array(modes)[:, bsz][0] == 'txt':
                first_modal = txt_seq_len - 1
                logit[bsz, :first_modal, self.num_txt_tokens:] = max_neg_value
                logit[bsz, first_modal:, :self.num_txt_tokens] = max_neg_value
            else:
                first_modal = img_seq_len - 1
                if np.array(modes)[:, bsz][1] == 'txt':
                    logit[bsz, :first_modal, :self.num_txt_tokens] = max_neg_value
                    logit[bsz, first_modal: (first_modal + txt_seq_len), self.num_txt_tokens:] = max_neg_value
                    logit[bsz, (first_modal + txt_seq_len):, :self.num_txt_tokens] = max_neg_value
                elif np.array(modes)[:, bsz][-1] == 'txt':
                    logit[bsz, :-txt_seq_len, :self.num_txt_tokens] = max_neg_value
                    logit[bsz, -txt_seq_len:, self.num_txt_tokens:] = max_neg_value
                if 'img3' in batch.keys() and np.array(modes)[:, bsz][2] == 'txt':  # [i, i, t, i]
                    logit[bsz, :(first_modal + img_seq_len), :self.num_txt_tokens] = max_neg_value
                    logit[bsz, (first_modal + img_seq_len):(first_modal + img_seq_len + txt_seq_len), self.num_txt_tokens:] = max_neg_value
                    logit[bsz, -img_seq_len:, :self.num_txt_tokens] = max_neg_value

        logit = logit.reshape(-1, logit.size(-1))

        target_lst = []
        for bsz in range(batch_size):
            for idx, mode in enumerate(np.array(modes)[:, bsz]):
                if idx == 0:
                    target = batch[mode][bsz, 1:]
                else:
                    target = batch[mode][bsz]
                if mode.startswith('img'):
                    target_lst.append(target + self.num_txt_tokens)
                else:
                    target_lst.append(target)
        target = torch.cat(target_lst, dim=0)

        ignore_classes = torch.ones(self.num_txt_tokens + self.num_img_tokens)
        ignore_classes[1024 + self.num_txt_tokens] = 0.
        loss = cross_entropy(logit, target, ignore_index=self.pad_token_idx, weight=ignore_classes.to(logit.device))

        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True)

        output = {
            'batch_idx': batch_idx,
            'loss': loss
        }
        return output

    def on_train_epoch_end(self): # updated due to dependency; def training_epoch_end(self, outputs):
        torch.cuda.empty_cache()

    def on_test_start(self):
        print("🧹 Clearing previous test outputs at the beginning of test!", flush=True)
        self.test_outputs = []

    def test_step(self, batch, batch_idx):
        print(f"\n🚥 Running test_step for batch {batch_idx}", flush=True)
        img_paths = batch['img_paths']
        study_ids = batch['study_id']
        print(f"📄 Studies in batch: {study_ids}", flush=True)
        self.transformerLM_unified.eval()

        img1 = batch['img1']
        txt = batch['txt']
        view = batch['view_position']
        self.transformerLM_unified.max_img_num = self.max_img_num

        def setup_modes(batch, max_img_num, random_mode_order=False):
            """
            If random_mode_order=False → fixed, deterministic order.
            If random_mode_order=True  → sample exactly as original code did.
            Returns: modes_txt, modes_img1, modes_img2 (or None), modes_img3 (or None)
            """
            print(f"random_order_setup: {random_mode_order}")
            if max_img_num == 1:
                    modes_txt  = [['img1'], ['txt']]
                    modes_img1 = [['txt'], ['img1']]
                    return modes_txt, modes_img1, None, None

            # for 2 images
            if max_img_num == 2:
                # two possible text orders:
                txt_orders = [
                    [['img1'], ['img2'], ['txt']],
                    [['img2'], ['img1'], ['txt']]
                ]
                img1_orders = [
                    [['img2'], ['txt'], ['img1']],
                    [['txt'],  ['img2'], ['img1']],
                ]
                img2_orders = [
                    [['img1'], ['txt'], ['img2']],
                    [['txt'],  ['img1'], ['img2']],
                ]

                if random_mode_order:
                    modes_txt  = random.choice(txt_orders)
                    modes_img1 = random.choice(img1_orders)
                    modes_img2 = random.choice(img2_orders)
                else:
                    # pick the first entry of each list for a fixed run
                    modes_txt, modes_img1, modes_img2 = txt_orders[0], img1_orders[0], img2_orders[0]

                return modes_txt, modes_img1, modes_img2, None

            # for 3 images
            if max_img_num == 3:
                txt_choices = [
                    [['img1'], ['img2'], ['img3'], ['txt']],
                ]
                img1_choices = [
                    [['txt'], ['img2'], ['img3'], ['img1']],
                ]
                img2_choices = [
                    [['txt'], ['img1'], ['img3'], ['img2']],
                ]
                img3_choices = [
                    [['txt'], ['img1'], ['img2'], ['img3']],
                ]
                # In the original you random.sample permutations of 3-element lists then append the fourth.
                base_txt_modes  = [['img1'], ['img2'], ['img3']]
                base_img1_modes = [['txt'],  ['img2'], ['img3']]
                base_img2_modes = [['txt'],  ['img1'], ['img3']]
                base_img3_modes = [['txt'],  ['img1'], ['img2']]

                if random_mode_order:
                    modes_txt  = random.sample(base_txt_modes, 3) + [['txt']]
                    modes_img1 = random.sample(base_img1_modes, 3) + [['img1']]
                    modes_img2 = random.sample(base_img2_modes, 3) + [['img2']]
                    modes_img3 = random.sample(base_img3_modes, 3) + [['img3']]
                else:
                    modes_txt, modes_img1, modes_img2, modes_img3 = \
                      [['img1'], ['img2'], ['img3'], ['txt']], \
                      [['txt'],  ['img2'], ['img3'], ['img1']], \
                      [['txt'],  ['img1'], ['img3'], ['img2']], \
                      [['txt'],  ['img1'], ['img2'], ['img3']]

                return modes_txt, modes_img1, modes_img2, modes_img3

            raise ValueError(f"Unsupported max_img_num={max_img_num}")

        random_mode = self.hparams.get('random_mode_order', True)
        modes_txt, modes_img1, modes_img2, modes_img3 = setup_modes(batch, self.max_img_num, random_mode_order=True) ## Set random_mode_order=False for fixed token order
        with torch.no_grad():
            # Text generation
            batch['modes'] = modes_txt
            start = time.time()
            print("🧠 Generating text...", flush=True)
            gen_texts = self.transformerLM_unified.generate_texts(
                batch,
                sos_token_idx=self.sos_token_idx,
                eos_token_idx=self.eos_token_idx,
                pad_token_idx=self.pad_token_idx,
                filter_logits_fn='top_p',
                filter_thres=0.9,
                temperature=0.7,
                causal=self.causal
            )
            print(f"🕓 Text generated in {time.time() - start:.2f}s", flush=True)

            # Image 1 generation
            batch['modes'] = modes_img1
            start = time.time()
            print("🧠 Generating image1...", flush=True)
            gen_images1 = self.transformerLM_unified.generate_image(
                batch,
                filter_logits_fn='top_p',
                filter_thres=0.9,
                temperature=0.7,
                causal=self.causal
            )
            print(f"🕓 Image1 generated in {time.time() - start:.2f}s", flush=True)

            # image2
            if 'img2' in batch:
                batch['modes'] = modes_img2
                start = time.time()
                print("🧠 Generating image2...", flush=True)
                gen_images2 = self.transformerLM_unified.generate_image(batch, filter_logits_fn='top_p',
                                                                      filter_thres=0.9, temperature=0.7,
                                                                      causal=self.causal)
                print(f"🕓 Image2 generated in {time.time() - start:.2f}s", flush=True)

            # image3
            if 'img3' in batch:
                batch['modes'] = modes_img3
                start = time.time()
                print("🧠 Generating image3...", flush=True)
                gen_images3 = self.transformerLM_unified.generate_image(batch, filter_logits_fn='top_p',
                                                                      filter_thres=0.9, temperature=0.7,
                                                                      causal=self.causal)
                print(f"🕓 Image3 generated in {time.time() - start:.2f}s", flush=True)

        output = {
            'GT_text': txt,
            'gen_text': gen_texts,
            'GT_image1': img1,
            'gen_image1': gen_images1,
            'img_paths': img_paths,
            'modes_txt': modes_txt,
            'modes_img1': modes_img1,
            'view': view,
        }

        if 'img2' in batch:
            output['GT_image2'] = batch['img2']
            output['gen_image2'] = gen_images2
            output['modes_img2'] = modes_img2

        if 'img3' in batch:
            output['GT_image3'] = batch['img3']
            output['gen_image3'] = gen_images3
            output['modes_img3'] = modes_img3

        self.test_outputs.append(output)
        print(f"✅ Finished batch {batch_idx}", flush=True)
        return output


    ### Added
    def on_test_epoch_end(self):
        if not self.test_outputs:
            print("⚠️ Warning: No test outputs gathered!")
            return

        test_step_outputs = self.test_outputs  # retrieve stored outputs
        print(f"✅ Processing {len(test_step_outputs)} gathered test batches...", flush=True)

        from tokenizers import ByteLevelBPETokenizer
        from tokenizers.processors import BertProcessing

        tokenizer = ByteLevelBPETokenizer('BBPE_tokenizer/vocab.json', 'BBPE_tokenizer/merges.txt')
        tokenizer.add_special_tokens(["[PAD]", "[SOS]", "[EOS]", "[SEP]", "[MASK]"])
        tokenizer._tokenizer.post_processor = BertProcessing(
            ("[EOS]", tokenizer.token_to_id("[EOS]")),
            ("[SOS]", tokenizer.token_to_id("[SOS]")),
        )
        tokenizer.enable_truncation(max_length=256)
        tokenizer.enable_padding(pad_id=tokenizer.token_to_id("[PAD]"), pad_token="[PAD]", length=256)
        if self.global_rank == 0:
            img_paths = self.test_outputs[0]['img_paths']
            if self.max_img_num != -1:
                max_text_len = self.test_outputs[0]['GT_text'].size(-1)
                total_GT_text = torch.empty(0, max_text_len).type_as(self.test_outputs[0]['GT_text'])
                total_gen_text = torch.empty(0, max_text_len).type_as(self.test_outputs[0]['GT_text'])

                for out in self.test_outputs:
                    GT_text = out['GT_text'].reshape(-1, max_text_len)
                    gen_text = out['gen_text'].reshape(-1, max_text_len)
                    total_GT_text = torch.cat((total_GT_text, GT_text), dim=0)
                    total_gen_text = torch.cat((total_gen_text, gen_text), dim=0)

            ckpt_name = self.ckpt_path.split('/')[-1].split('-')[0] if self.ckpt_path else "no_ckpt"
            output_path = os.path.join(
                self.save_dir,
                f"test_output_{ckpt_name}_{str(self.max_img_num)}_of_{str(self.target_count)}_{self.test_meta_file_name}.pt"
            )
            os.makedirs(self.save_dir, exist_ok=True)   ## added directory check
            torch.save(self.test_outputs, output_path)
            print(f"✅ Test outputs saved at {output_path}")

            # Decode for BLEU score
            GT_decoded_texts, gen_decoded_texts = [], []
            for gt_text_i, gen_text_i in zip(total_GT_text, total_gen_text):
                gt_decoded_text_i = tokenizer.decode(gt_text_i.tolist(), skip_special_tokens=True)
                gen_decoded_text_i = tokenizer.decode(gen_text_i.tolist(), skip_special_tokens=True)
                GT_decoded_texts.append(gt_decoded_text_i)
                gen_decoded_texts.append(gen_decoded_text_i)

            from nltk.translate.bleu_score import corpus_bleu

            references = [[text.split()] for text in GT_decoded_texts]
            candidates = [text.split() for text in gen_decoded_texts]

            bleu1 = corpus_bleu(references, candidates, weights=(1, 0, 0, 0))
            bleu2 = corpus_bleu(references, candidates, weights=(0.5, 0.5, 0, 0))
            bleu3 = corpus_bleu(references, candidates, weights=(1/3, 1/3, 1/3, 0))
            bleu4 = corpus_bleu(references, candidates, weights=(0.25, 0.25, 0.25, 0.25))

            print(f"BLEU-1: {bleu1:.3f}")
            print(f"BLEU-2: {bleu2:.3f}")
            print(f"BLEU-3: {bleu3:.3f}")
            print(f"BLEU-4: {bleu4:.3f}")

            self.log("test_BLEU-1", bleu1)
            self.log("test_BLEU-2", bleu2)
            self.log("test_BLEU-3", bleu3)
            self.log("test_BLEU-4", bleu4)

        self.test_outputs = []  # Clear outputs for safety


    # def configure_optimizers(self):
    #     optimizer = AdamW(self.parameters(), lr=self.lr)
    #     train_loader = self.train_dataloader()
    #     scheduler = {
    #         'scheduler':
    #             get_cosine_schedule_with_warmup(
    #                 optimizer=optimizer,
    #                 num_warmup_steps=0,
    #                 num_training_steps=self.kargs['epochs'] * len(train_loader)),
    #         'interval': 'step',
    #     }
    #     return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def configure_optimizers(self):
      optimizer = AdamW(self.parameters(), lr=self.lr)
      
      # Safe fallback if you don't have exact num_training_steps:
      # You can pass total_steps via kargs OR adjust here manually.
      num_training_steps = self.kargs.get('total_steps', self.kargs['epochs'] * 1000)  # <-- replace 1000 with an estimate
      
      scheduler = {
          'scheduler': get_cosine_schedule_with_warmup(
              optimizer=optimizer,
              num_warmup_steps=0,
              num_training_steps=num_training_steps
          ),
          'interval': 'step',
      }
      return {"optimizer": optimizer, "lr_scheduler": scheduler}
