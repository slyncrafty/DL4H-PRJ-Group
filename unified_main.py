import os
import argparse
import datetime
from functools import partial
import torch.nn as nn
import torch
torch.set_float32_matmul_precision('high')
print(f"✅ Precision setting: {torch.get_float32_matmul_precision()}")
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing  

from helpers import str2bool
from datamodule import CXRDataModule
from loader_unified import UnifiedCXRDataset
from unified_plmodel import TransformerLightning_unified

print("🚀 Starting UniXGen Training Script", flush=True)

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42, type=int)

    parser.add_argument('--test', type=str2bool, default=False, help='trian (False) or test (True)')
    parser.add_argument('--reload_ckpt_dir', default='ckpt/unixgen_lightning.ckpt', type=str, help='ckpt_dir')

    parser.add_argument('--n_gpus', default=1, type=int)
    parser.add_argument('--n_epochs', default=200, type=int)
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--lr', default=4.5e-6, type=float, help='learning rate')
    parser.add_argument('--accumulate_grad_batches', default=1, type=float)
    parser.add_argument('--weight_decay', default=1e-6, type=float, help='weight decay')

    parser.add_argument('--img_root_dir', default='data/images', type=str)
    parser.add_argument('--text_root_dir', default='data/reports', type=str)
    parser.add_argument('--train_meta_file', default='metadata/mimiccxr_train_sub_final.csv', type=str)
    parser.add_argument('--val_meta_file', default='metadata/mimiccxr_validate_sub_final.csv', type=str)
    parser.add_argument('--test_meta_file', default='metadata/mimiccxr_test_sub_final.csv', type=str)
    parser.add_argument('--vocab_file', default='BBPE_tokenizer/vocab.json', type=str)
    parser.add_argument('--merge_file', default='BBPE_tokenizer/merges.txt', type=str)

    parser.add_argument('--vqgan', default=512, type=int, help='vqgan img resolution')
    parser.add_argument('--vqgan_model_path', default='mimiccxr_vqgan/last.ckpt', type=str)
    parser.add_argument('--vqgan_config_path', default='mimiccxr_vqgan/2021-12-17T08-58-54-project.yaml', type=str)
    parser.add_argument('--codebook_indices_path', default='mimiccxr_vqgan/mimiccxr_vqgan1024_res512_codebook_indices.pickle', type=str)


    parser.add_argument('--max_img_num', default=3, type=int, help='must be less than or equal to target_count')
    parser.add_argument('--target_count', default=3, type=int, help='select target goup, S w/1, w/2, w/3')
    parser.add_argument('--under_sample', default='fixed_all_unified', type=str)
    parser.add_argument('--max_text_len', default=256, type=int)
    parser.add_argument('--target_view', default=['AP', 'PA', 'LATERAL', 'LL'], nargs='+', type=str)

    parser.add_argument('--transformer', default=True)
    parser.add_argument('--FAVOR', default=True)
    parser.add_argument('--generalized_attention', default=True, help='defaults to softmax approximation, but can be set to True for generalized attention')
    parser.add_argument('--dim', default=768, type=int, help='dimension. dimension must be divisible by number of heads.')
    parser.add_argument('--depth', default=12, type=int, help='layers')
    parser.add_argument('--heads', default=12, type=int, help='heads')
    parser.add_argument('--layer_norm_eps', default=1e-12, type=float)
    parser.add_argument('--dim_head', default=64, type=int, help='dim of head. inner_dim = dim_head * heads')
    parser.add_argument('--local_attn_heads', default=0, type=int, help='if n heads are local attention, heads-n others are global performers.')
    parser.add_argument('--local_window_size', default=256, type=int, help='window size of local attention')
    parser.add_argument('--causal', default=True, type=str2bool, help='auto-regressive or not')
    parser.add_argument('--attn_type', default='all_modality_causal_cuda', type=str)
    parser.add_argument('--causal_clm', default='conditioned_causal', choices=['conditioned_causal', 'causal'], type=str, help='Not in used')
    parser.add_argument('--nb_features', default=64, type=int, help='number of random features, if not set, will default to (d * log(d)), where d is the dimension of each head.')
    parser.add_argument('--feature_redraw_interval', default=1000, type=int,
                        help='how frequently to redraw the projection matrix, the more frequent, the slower the training')
    parser.add_argument('--reversible', default=False, type=str2bool,
                        help='reversible layers, from Reformer paper. Works only when sharded_ddp=True')
    parser.add_argument('--ff_chunks', default=1, type=int, help='chunk feedforward layer, from Reformer paper')
    parser.add_argument('--ff_glu', default=False, type=str2bool, help='use GLU variant for feedforward')
    parser.add_argument('--emb_dropout', default=0.1, type=float, help='embedding dropout')
    parser.add_argument('--ff_dropout', default=0.1, type=float, help='feedforward dropout')
    parser.add_argument('--attn_dropout', default=0.1, type=float, help='post-attn dropout')
    parser.add_argument('--use_scalenorm', default=False, type=str2bool,
                        help='use scale norm, from Transformers without Tears paper')
    parser.add_argument('--use_rezero', default=False, type=str2bool,
                        help='use rezero, from Rezero is all you need paper')
    parser.add_argument('--tie_embed', default=False, type=str2bool,
                        help='multiply final embeddings with token weights for logits')
    parser.add_argument('--rotary_position_emb', default=False, type=str2bool,
                        help='use rotary positional embedding, which endows linear attention with relative positional encoding with no learned parameters. should always be turned on unless if you want to go back to old absolute positional encoding')

    parser.add_argument('--save_top_k', default=1, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--gradient_clip_val', default=0, type=float)
    parser.add_argument('--num_sanity_val_steps', default=0, type=int)
    parser.add_argument('--fp16', default=False, type=str2bool, help='FP16')
    parser.add_argument('--sharded_ddp', default=False, type=str2bool, help='fairscale sharded ddp')

    parser.add_argument('--random_mode_order', default=True, type=str2bool, help='Use random order for setup_modes()')
    parser.add_argument('--save_dir', default='output', type=str, help='Where to write output .pt')
    parser.add_argument('--limit_test_batches', default=1.0, type=float, help='Portion (or count) of test‐batches to run: 1.0=all, 0.1=10%, int=that many batches')
    
    args = parser.parse_args()

    print(f"⚙️ Parsed Arguments: {args}", flush=True)
    
    start = datetime.datetime.now()
    print(f"🕒 Script started at {start}", flush=True)

    pl.seed_everything(args.seed, workers=True)
    print("🌱 Set global seed.", flush=True)

    if args.test:                                      # inference run
    # 1. peek into the checkpoint hyper-parameters
        ckpt_meta = torch.load(args.reload_ckpt_dir, map_location='cpu')["hyper_parameters"]
        for key in ("max_seq_len", "num_img_tokens", "img_vocab_size",
                    "num_img_tokens", "img_len","img_fmap_size", 
                    "num_tokens", "max_img_len"):
            if key in ckpt_meta:
                setattr(args, key, ckpt_meta[key])
            else:
                print(f"⚠️⚠️⚠️ Warning: Key '{key}' not found in checkpoint metadata. ⚠️⚠️⚠️")
            
        # # Fallback only if not explicitly passed
        # if not args.max_img_num or args.max_img_num < 1:
        #     args.max_img_num = ckpt_meta.get('max_img_num', 1)
        # if not args.target_count or args.target_count < 1:
        #     args.target_count = ckpt_meta.get('target_count', 1)
    print("🛠️ Initializing Tokenizer...", flush=True)
    tokenizer = ByteLevelBPETokenizer(
        args.vocab_file,
        args.merge_file,
    )
    # ─── Sanity check: do vocab & checkpoint  ──────────
    print("\n🧪  Tokenizer sanity check:")
    first_30 = list(range(30))
    decoded  = tokenizer.decode(first_30, skip_special_tokens=True)
    print("First 30 token IDs →", decoded[:120], "...\n")
    # Optional: assert on a known token if authors documented one
    assert "[PAD]" in tokenizer.get_vocab(), "Tokenizer does not have PAD token!"

    tokenizer.add_special_tokens(["[PAD]", "[SOS]", "[EOS]", "[SEP]", "[MASK]"])
    tokenizer._tokenizer.post_processor = BertProcessing(
        ("[EOS]", tokenizer.token_to_id("[EOS]")),
        ("[SOS]", tokenizer.token_to_id("[SOS]")),
    )
    tokenizer.enable_truncation(max_length=args.max_text_len)
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id("[PAD]"), pad_token="[PAD]", length=args.max_text_len)  
    print("✅ Tokenizer initialized.", flush=True)

    print("📁 Setting up Datasets...", flush=True)
    dsclass = partial(
        UnifiedCXRDataset,
        img_root_dir=args.img_root_dir,
        text_root_dir=args.text_root_dir,
        vqgan_model_path=args.vqgan_model_path,
        vqgan_config_path=args.vqgan_config_path,
        codebook_indices_path=args.codebook_indices_path,
        vqgan=args.vqgan,
        max_img_num=args.max_img_num,
        max_text_len=args.max_text_len,
        tokenizer=tokenizer,
        target_count=args.target_count,
        target_view=args.target_view,
        under_sample=args.under_sample,
    )

    print("📌 Loading Train Dataset...", flush=True)
    train_ds = dsclass(args.train_meta_file)
    print("📌 Loading Validation Dataset...", flush=True)
    val_ds = dsclass(args.val_meta_file)
    print("📌 Loading Test Dataset...", flush=True)
    test_ds = dsclass(args.test_meta_file)
    print("✅ Datasets loaded successfully.", flush=True)

    dm = CXRDataModule(
        train_ds, val_ds, test_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    import math ##
    num_samples = len(train_ds)## added
    steps_per_epoch = math.ceil(num_samples / args.batch_size)## added
    total_steps = args.n_epochs * steps_per_epoch    ## added
    print("✅ DataModule initialized.", flush=True)

    if not args.test:
        args.num_tokens = train_ds.text_vocab_size
        args.img_vocab_size = train_ds.img_vocab_size
        args.num_img_tokens = train_ds.img_vocab_size + (2*4) + args.max_img_num
        args.max_text_len = train_ds.max_text_len  
        args.max_img_len = train_ds.img_len * args.max_img_num
        args.max_seq_len = args.max_img_len + args.max_text_len
        args.img_len = train_ds.img_len
        args.img_fmap_size = int(train_ds.img_fmap_size)

    print(f"🧮 max_seq_len: {args.max_seq_len}, num_tokens (text): {args.num_tokens}, img_vocab_size: {args.img_vocab_size}", flush=True)

    kargs_unified = {
        'num_tokens': args.num_tokens,
        'num_img_tokens': args.num_img_tokens,
        'img_vocab_size': args.img_vocab_size, 
        'max_seq_len': args.max_seq_len,
        'max_img_len': args.max_img_len,
        'max_img_num': args.max_img_num,
        'img_len': args.img_len,
        'dim': args.dim,
        'depth': args.depth,
        'heads': args.heads,
        'dim_head': args.dim_head,
        'local_attn_heads': args.local_attn_heads,
        'local_window_size': args.local_window_size,
        'causal': args.causal,
        'attn_type': args.attn_type,
        'nb_features': args.nb_features,
        'feature_redraw_interval': args.feature_redraw_interval,
        'reversible': args.reversible,
        'ff_chunks': args.ff_chunks,
        'ff_glu': args.ff_glu,
        'emb_dropout': args.emb_dropout,
        'ff_dropout': args.ff_dropout,
        'attn_dropout': args.attn_dropout,
        'generalized_attention': args.generalized_attention,
        'kernel_fn': nn.ReLU(),
        'use_scalenorm': args.use_scalenorm,
        'use_rezero': args.use_rezero,
        'tie_embed': args.tie_embed,
        'rotary_position_emb': args.rotary_position_emb,
        'img_fmap_size': args.img_fmap_size,
        'FAVOR': args.FAVOR,
        'epochs': args.n_epochs,
        'ckpt_dir': args.reload_ckpt_dir,
        'under_sample': args.under_sample,
        'target_count': args.target_count,
        'random_mode_order': args.random_mode_order,
    }

    print("🧩 Initializing Model...", flush=True)
    if args.test:                                      # inference run
        # 2. load the model directly from checkpoint (no prior instantiation)
        model = TransformerLightning_unified.load_from_checkpoint(
            args.reload_ckpt_dir,
            lr=args.lr,
            weight_decay=args.weight_decay,
            tokenizer=tokenizer,
            pad_token_idx=tokenizer.token_to_id("[PAD]"),
            sos_token_idx=tokenizer.token_to_id("[SOS]"),
            eos_token_idx=tokenizer.token_to_id("[EOS]"),
            save_dir=args.save_dir,
            causal_trans=args.causal_clm,
            total_steps=total_steps,
            **kargs_unified,
        )
        model.ckpt_path            = args.reload_ckpt_dir
        model.test_meta_file_name  = os.path.basename(args.test_meta_file).split('.')[0]

        model.max_img_num          = args.max_img_num
        model.hparams.max_img_num  = args.max_img_num  # ensure hyperparameters match
        model.target_count         = args.target_count
        model.hparams.target_count = args.target_count  # ensure hyperparameters match
        
        model.save_dir             = args.save_dir            # make sure it exists
        model.transformerLM_unified.max_img_num = args.max_img_num
        model.transformerLM_unified.target_count = args.target_count
        os.makedirs(model.save_dir, exist_ok=True)
        print("✅ Checkpoint loaded with matching shapes.", flush=True)

    else:                                              # training run from scratch
        model = TransformerLightning_unified(
            lr=args.lr,
            weight_decay=args.weight_decay,
            tokenizer=tokenizer,
            pad_token_idx=tokenizer.token_to_id("[PAD]"),
            sos_token_idx=tokenizer.token_to_id("[SOS]"),
            eos_token_idx=tokenizer.token_to_id("[EOS]"),
            save_dir=args.save_dir,
            causal_trans=args.causal_clm,
            total_steps=total_steps,
            **kargs_unified,
        )
        print("✅ Fresh model initialised.", flush=True)
    if not args.test:
        # ─── quick sanity loss --------------------------------------------------
        batch = next(iter(dm.val_dataloader()))
        loss  = model.training_step(batch, 0)['loss'].item()
        print(f"🧪  CE loss on one val batch: {loss:.3f}")   # <≈5 means weights OK
        # -----------------------------------------------------------------------
        print("🔑 token-emb slice:", model.transformerLM_unified.token_emb.weight[0,:5])

    checkpoint_callback = ModelCheckpoint(
        dirpath='output',
        filename='{epoch:02d}-{train_loss:.2f}',
        verbose=True,
        save_last=True,
        save_top_k=int(args.n_epochs / args.save_top_k),
        every_n_epochs=args.save_top_k,
        monitor='train_loss',
        mode='min',
    )

    lr_callback = LearningRateMonitor(logging_interval="step")
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=True, mode='max')

    trainer_args = {
        'callbacks': [checkpoint_callback, lr_callback],
        'max_epochs': args.n_epochs,
        # 'gpus': args.n_gpus,
        'accelerator': 'gpu', 
        'devices': args.n_gpus,
        'num_sanity_val_steps': args.num_sanity_val_steps,
        'log_every_n_steps': 1,
        # 'terminate_on_nan': True,
        'detect_anomaly': True,
        #'checkpoint_callback': True,
        'enable_checkpointing': True,
        # 'ckpt_path': args.reload_ckpt_dir,
    }

    print("🎯 Preparing Trainer...", flush=True)

    wandb_logger = WandbLogger(name=str(datetime.datetime.now()), log_model=True, config=args, save_code=True)

    trainer = pl.Trainer(**trainer_args, logger=wandb_logger, strategy=DDPStrategy(find_unused_parameters=True),
                         gradient_clip_val=args.gradient_clip_val, profiler="simple",
                         accumulate_grad_batches=args.accumulate_grad_batches,
                         #replace_sampler_ddp=False
                         limit_test_batches=args.limit_test_batches, ## 1.0 for 100%, 0.1 for 10%, int for batch counts
                         )

    print("🚦 Trainer initialized, starting training/testing...", flush=True)

    if not args.test:
        print("🚀 Starting Training...", flush=True)
        trainer.fit(model, datamodule=dm)
        print("✅ Training Completed.", flush=True)
    else:
        print("🧪 Starting Testing...", flush=True)
        # trainer.test(model, dataloaders=dm)
        print(f"[DEBUG] Loaded max_img_num: {model.max_img_num} | transformer: {model.transformerLM_unified.max_img_num}")
        print(f"[DEBUG] Loaded target_count: {model.target_count} | transformer: {model.transformerLM_unified.target_count}")
        trainer.test(model, dataloaders=dm.test_dataloader()) ## pytorch lightning v2.x fix
        print("✅ Testing Completed.", flush=True)
