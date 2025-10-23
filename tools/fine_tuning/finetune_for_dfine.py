#!/usr/bin/env python3
"""
Fine-tune MOTRv2's yolox_embed layer for D-FINE compatibility.

This script freezes all model parameters except yolox_embed and trains
it to adapt to D-FINE's detection characteristics.

Usage:
    python finetune_for_dfine.py @configs/beach_volleyball.args --resume motrv2.pth
"""

import argparse
import datetime
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from util.tool import load_model
import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset
from engine import train_one_epoch_mot
from models import build_model
from main import get_args_parser


def freeze_except_yolox_embed(model):
    """
    Freeze all parameters except yolox_embed for fine-tuning.

    This allows the model to adapt to D-FINE detections without
    forgetting its tracking abilities.
    """
    print("\n" + "="*80)
    print("Setting up partial training: yolox_embed ONLY")
    print("="*80 + "\n")

    trainable_params = []
    frozen_params = []

    for name, param in model.named_parameters():
        if 'yolox_embed' in name:
            param.requires_grad = True
            trainable_params.append((name, param.numel()))
            print(f"✓ TRAINING: {name:50s} shape={list(param.shape)}")
        else:
            param.requires_grad = False
            frozen_params.append(param.numel())

    # Statistics
    total_trainable = sum(p[1] for p in trainable_params)
    total_frozen = sum(frozen_params)
    total_params = total_trainable + total_frozen

    print(f"\n{'='*80}")
    print(f"Parameter Summary:")
    print(f"  Trainable:  {total_trainable:>12,} ({100*total_trainable/total_params:>5.2f}%)")
    print(f"  Frozen:     {total_frozen:>12,} ({100*total_frozen/total_params:>5.2f}%)")
    print(f"  Total:      {total_params:>12,}")
    print(f"{'='*80}\n")

    return model


def freeze_more_layers(model, train_layers=['yolox_embed', 'track_embed', 'query_interaction']):
    """
    Freeze all except specified layers (for Option 2 - more aggressive fine-tuning).

    Args:
        train_layers: List of layer name keywords to train
    """
    print("\n" + "="*80)
    print(f"Setting up partial training: {', '.join(train_layers)}")
    print("="*80 + "\n")

    trainable_params = []
    frozen_params = []

    for name, param in model.named_parameters():
        should_train = any(layer in name for layer in train_layers)

        if should_train:
            param.requires_grad = True
            trainable_params.append((name, param.numel()))
            print(f"✓ TRAINING: {name:50s} shape={list(param.shape)}")
        else:
            param.requires_grad = False
            frozen_params.append(param.numel())

    # Statistics
    total_trainable = sum(p[1] for p in trainable_params)
    total_frozen = sum(frozen_params)
    total_params = total_trainable + total_frozen

    print(f"\n{'='*80}")
    print(f"Parameter Summary:")
    print(f"  Trainable:  {total_trainable:>12,} ({100*total_trainable/total_params:>5.2f}%)")
    print(f"  Frozen:     {total_frozen:>12,} ({100*total_frozen/total_params:>5.2f}%)")
    print(f"  Total:      {total_params:>12,}")
    print(f"{'='*80}\n")

    return model


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    print("\n" + "="*80)
    print("D-FINE ADAPTATION FINE-TUNING")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Detection DB: {args.det_db}")
    print(f"  Resume from: {args.resume}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Training strategy: {'embed_only' if args.embed_only else 'multi_layer'}")
    print("="*80 + "\n")

    device = torch.device(args.device)

    # Fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Build model
    model, criterion, postprocessors = build_model(args)
    model.to(device)

    # Load pretrained checkpoint
    if args.resume:
        print(f"\n📥 Loading checkpoint from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')

        # Try to load model state
        if 'model' in checkpoint:
            model_state = checkpoint['model']
        else:
            model_state = checkpoint

        missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)

        if missing_keys:
            print(f"⚠️  Missing keys: {len(missing_keys)}")
            for key in missing_keys[:5]:  # Show first 5
                print(f"    - {key}")
        if unexpected_keys:
            print(f"⚠️  Unexpected keys: {len(unexpected_keys)}")
            for key in unexpected_keys[:5]:
                print(f"    - {key}")

        print("✓ Checkpoint loaded successfully\n")
    else:
        print("⚠️  WARNING: No checkpoint specified! Training from scratch.\n")

    # Apply freezing strategy
    if args.embed_only:
        model = freeze_except_yolox_embed(model)
    else:
        train_layers = ['yolox_embed', 'track_embed', 'query_interaction']
        model = freeze_more_layers(model, train_layers)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\n📊 Total trainable parameters: {n_parameters:,}\n')

    # Build dataset
    dataset_train = build_dataset(image_set='train', args=args)
    print(f"📦 Training dataset size: {len(dataset_train)} samples\n")

    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
        else:
            sampler_train = samplers.DistributedSampler(dataset_train)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    collate_fn = utils.mot_collate_fn
    data_loader_train = DataLoader(
        dataset_train,
        batch_sampler=batch_sampler_train,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Setup optimizer (only for trainable parameters)
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if args.sgd:
        optimizer = torch.optim.SGD(
            trainable_params,
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay
        )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    output_dir = Path(args.output_dir)

    print("="*80)
    print("🚀 Starting fine-tuning")
    print("="*80 + "\n")

    start_time = time.time()

    dataset_train.set_epoch(args.start_epoch)

    for epoch in range(args.start_epoch, args.epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*80}\n")

        if args.distributed:
            sampler_train.set_epoch(epoch)

        train_stats = train_one_epoch_mot(
            model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm
        )

        lr_scheduler.step()

        # Save checkpoint
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']

            # Save every epoch for fine-tuning (cheap since only yolox_embed changes)
            checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')

            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

            print(f"\n✓ Checkpoint saved to: {checkpoint_path}")

        dataset_train.step_epoch()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    print("\n" + "="*80)
    print(f"✅ Fine-tuning complete!")
    print(f"   Total time: {total_time_str}")
    print(f"   Final checkpoint: {output_dir / 'checkpoint.pth'}")
    print("="*80 + "\n")

    print("📋 Next steps:")
    print("   1. Test with: python submit_dance.py --resume outputs/.../checkpoint.pth")
    print("   2. Compare tracking performance with original model")
    print("   3. If needed, try fine-tuning more layers (set --embed_only False)\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'MOTRv2 fine-tuning for D-FINE',
        parents=[get_args_parser()],
        add_help=False
    )

    # Add fine-tuning specific arguments
    parser.add_argument('--embed_only', default=True, type=lambda x: str(x).lower() == 'true',
                       help='If True, only train yolox_embed. If False, train multiple layers.')

    args = parser.parse_args()

    # Validate arguments
    if not args.resume:
        print("\n⚠️  WARNING: --resume not specified!")
        print("    You should provide a pretrained MOTRv2 checkpoint.")
        print("    Example: --resume motrv2_checkpoint.pth\n")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            exit(0)

    if not args.det_db:
        print("\n❌ ERROR: --det_db must be specified!")
        print("   This should point to your D-FINE detection database.")
        print("   Example: --det_db det_db_beach_volleyball.json\n")
        exit(1)

    # Ensure output directory exists
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
