#!/usr/bin/env python
# Dementia detection from retinal fundus images — DDP trainer
#
# Trains a ViT backbone with a single-logit head (binary cross-entropy) for prevalent dementia detection using
# PyTorch DDP. Supports three adaptation strategies: linear probing, partial fine-tuning (last N blocks) and LoRA.
# Training runs for up to 80 epochs; the checkpoint with the highest validation AUROC is kept, the learning rate is
# halved after `lr_halve_patience` (5) epochs without improvement and training stops after `early_stop_patience` (10).
# The selected checkpoint is then applied, without further training, to the validation and test sets (AUROC,
# Youden threshold) and to the future-incidence cohort (Harrell's C-index), and per-image predictions are saved.
#
# Usage (example, 8 GPUs; settings per strategy in dementia.sh):
#   torchrun --nproc_per_node=8 trainer.py --arch dinov3_vith16plus --ft lora --ft_blks full --lora_rank 4 \
#       --lr 1e-4 --batch_size 8 --enable_amp
#
# Key arguments:
#   --arch      : backbone (retfound, retfound_dinov2, mae, openclip, dinov2, dinov3, dinov3_vith16plus)
#   --ft        : adaptation strategy (linear, partial, lora)
#   --det_csv   : dementia_detection.csv from data_preparation.ipynb (pngfilename, label, train/valid/test flags)
#   --pred_csv  : dementia_prediction.csv from data_preparation.ipynb (pngfilename, event, obs_time)
#   --root_dir  : output directory with one folder per configuration (<root_dir>/<desc>/)
#   --eval_only : skip training and run inference with an existing checkpoint
#
# Outputs (<root_dir>/<desc>/):
#   ckpt.pth.tar, valid_preds.csv, test_preds.csv, prediction_preds.csv, res.csv

import warnings
warnings.filterwarnings('ignore')
import os
import math
from pathlib import Path
import argparse
from functools import partial

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.utils.data as udata
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
import pandas as pd
from tqdm import trange, tqdm
from sklearn.metrics import roc_auc_score, roc_curve
from lifelines.utils import concordance_index

from data import DementiaDetectionDataset, DementiaPredictionDataset
from models.encoder import build_model


@torch.no_grad()
def concat_all_gather(tensor):
    tensors_gather = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output


class Trainer:
    def __init__(self, args):
        self._init_distributed_mode()
        self.args = args
        self.desc = f'{args.arch}_{args.ft}'
        if args.ft == 'lora':
            self.desc += f'_rank_{args.lora_rank}_ft_{args.ft_blks}'
        elif args.ft == 'partial':
            self.desc += f'_ft_{args.ft_blks}'

        self._build_loader()
        self._build_model()
        self._build_optimizer()

        self._init_file()

    def _init_distributed_mode(self):
        dist.init_process_group(backend="nccl")
        dist.barrier()

        self.world_size = dist.get_world_size()
        self.global_rank = dist.get_rank()
        self.local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(self.local_rank)
        self.device = torch.cuda.current_device()

        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    def _build_loader(self):
        dist_sampler = partial(udata.distributed.DistributedSampler,
                               num_replicas=self.world_size, rank=self.global_rank)
        self.train_dataset = DementiaDetectionDataset(self.args.det_csv, 'train', self.args.img_size)
        self.train_sampler = dist_sampler(dataset=self.train_dataset, shuffle=True, drop_last=True)
        self.train_loader = udata.DataLoader(
            dataset=self.train_dataset,
            sampler=self.train_sampler,
            pin_memory=True,
            drop_last=True,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
        )
        self.valid_dataset = DementiaDetectionDataset(self.args.det_csv, 'valid', self.args.img_size)
        self.valid_sampler = dist_sampler(dataset=self.valid_dataset, shuffle=False)
        self.test_dataset = DementiaDetectionDataset(self.args.det_csv, 'test', self.args.img_size)
        self.test_sampler = dist_sampler(dataset=self.test_dataset, shuffle=False)
        self.valid_loader = udata.DataLoader(
            dataset=self.valid_dataset,
            sampler=self.valid_sampler,
            pin_memory=True,
            batch_size=8,
            num_workers=4,
        )
        self.test_loader = udata.DataLoader(
            dataset=self.test_dataset,
            sampler=self.test_sampler,
            pin_memory=False,
            batch_size=8,
            num_workers=4,
        )

        self.prediction_dataset = DementiaPredictionDataset(self.args.pred_csv, self.args.img_size)
        self.prediction_sampler = dist_sampler(dataset=self.prediction_dataset, shuffle=False)
        self.prediction_loader = udata.DataLoader(
            dataset=self.prediction_dataset,
            sampler=self.prediction_sampler,
            pin_memory=False,
            batch_size=8,
            num_workers=4,
        )

    def _build_model(self):
        self.model = DistributedDataParallel(
            build_model(self.args).cuda(),
            device_ids=[self.device]
        )
        self.criterion = nn.BCEWithLogitsLoss()

    def _build_optimizer(self):
        self.optimizer = optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.args.lr, weight_decay=self.args.weight_decay
        )
        self.enable_amp = self.args.enable_amp
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.enable_amp)

    def _init_file(self):
        root = Path(self.args.root_dir)
        self.log_dir = root/self.desc
        if (self.global_rank == 0) and (not self.log_dir.exists()):
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def save(self):
        if self.global_rank == 0:
            torch.save(self.model.module.state_dict(), self.log_dir/'ckpt.pth.tar')
        dist.barrier()

    def load(self):
        ckpt = torch.load(self.log_dir/'ckpt.pth.tar', map_location=f'cuda:{self.device}')
        self.model.module.load_state_dict(ckpt)

    def _dist_eval_order(self, n, batch_size):
        """
        Rebuilds the sample order produced by DistributedSampler(shuffle=False) + concat_all_gather.
        Returns `order`, where order[i] is the dataset index of the i-th gathered prediction
        (the sampler pads the dataset to a multiple of world_size by repeating it from the start).
        """
        W = self.world_size
        total = math.ceil(n / W) * W
        padded = list(range(n))
        pad = total - n
        if pad > 0:  # same repeat-padding as DistributedSampler
            padded += (list(range(n)) * math.ceil(pad / n))[:pad]
        per_rank = [padded[r::W] for r in range(W)]
        order = []
        n_batches = math.ceil(len(per_rank[0]) / batch_size)
        for b in range(n_batches):
            for r in range(W):
                order.extend(per_rank[r][b * batch_size:(b + 1) * batch_size])
        return order

    def _save_sample_preds(self, dataset_df, preds, batch_size, out_name):
        """Saves per-image predictions on rank 0 (duplicates from DDP padding removed)."""
        if self.global_rank != 0:
            return
        n = len(dataset_df)
        order = self._dist_eval_order(n, batch_size)
        sample_preds = [float('nan')] * n
        for pos, di in enumerate(order):
            sample_preds[di] = float(preds[pos])
        out = dataset_df.copy()
        out['pred'] = sample_preds
        out.to_csv(self.log_dir / out_name, index=False)

    def train_one_epoch(self, epoch):
        self.train_sampler.set_epoch(epoch)
        self.model.train()
        running_loss = torch.tensor([0.], device=self.device)

        for x, y in tqdm(self.train_loader, desc='train', total=len(self.train_loader), disable=self.global_rank!=0):
            self.optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda', enabled=self.enable_amp):
                x = x.cuda(self.local_rank, non_blocking=True)
                y = y.cuda(self.local_rank, non_blocking=True)

                out = self.model(x)
                loss = self.criterion(out.squeeze(), y)
                running_loss += loss.detach().clone().view(1)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

        mean_loss = (concat_all_gather(running_loss).mean() / len(self.train_loader)).item()
        if self.global_rank == 0:
            print(f'Epoch {epoch}: train loss {mean_loss:.3f}')

    @torch.no_grad()
    def valid_one_epoch(self, epoch=1, kind='valid', save_csv=False):
        self.model.eval()
        loader  = self.valid_loader if kind == 'valid' else self.test_loader
        sampler = self.valid_sampler if kind == 'valid' else self.test_sampler

        preds, targets = [], []
        sampler.set_epoch(0)

        for x, y in tqdm(loader, desc=kind, total=len(loader), disable=self.global_rank!=0):
            x = x.cuda(self.local_rank, non_blocking=True)
            y = y.cuda(self.local_rank, non_blocking=True)
            with torch.amp.autocast(device_type='cuda', enabled=self.enable_amp):
                p = torch.sigmoid(self.model(x))
            preds.append(concat_all_gather(p))
            targets.append(concat_all_gather(y))

        preds   = torch.cat(preds).detach().cpu().squeeze().numpy()
        targets = torch.cat(targets).detach().cpu().squeeze().numpy().astype(int)
        preds   = preds.reshape(-1)
        targets = targets.reshape(-1)

        # per-image predictions (inference only)
        if save_csv:
            self._save_sample_preds(
                loader.dataset.df, preds, loader.batch_size, f'{kind}_preds.csv'
            )

        try:
            auc = roc_auc_score(targets, preds)
        except ValueError:      # single-class labels
            auc = float('nan')

        if kind == 'valid':
            # threshold maximizing Youden's J
            try:
                fpr, tpr, thr = roc_curve(targets, preds)
                youden = tpr - fpr
                best_idx  = youden.argmax()
                best_thr  = float(thr[best_idx])
                best_sens = float(tpr[best_idx])
                best_spec = float(1.0 - fpr[best_idx])

                self.best_threshold_valid = best_thr
                self.best_sens_valid = best_sens
                self.best_spec_valid = best_spec

                if self.global_rank == 0:
                    print(f"\tvalid auc {auc:.3f}")
                    print(f"\tvalid Youden's J best threshold: {best_thr:.6f} "
                          f"(sensitivity={best_sens:.3f}, specificity={best_spec:.3f})")
            except ValueError:
                if self.global_rank == 0:
                    print(f"\tvalid auc {auc:.3f}")
                    print("\tvalid Youden's J: cannot compute (single-class labels)")
            return auc

        if self.global_rank == 0:
            print(f'\t{kind} auc {auc:.3f}')
        return auc

    @torch.no_grad()
    def prediction(self, save_csv=False):
        """
        Predicts on the future-incidence cohort, computes Harrell's C-index (model output as risk score) and the
        threshold maximizing Youden's J with incident events as binary labels.
        """
        self.model.eval()

        preds, events, times = [], [], []
        for x, e, t in tqdm(self.prediction_loader, desc='prediction', total=len(self.prediction_loader), disable=self.global_rank!=0):
            x = x.cuda(self.local_rank, non_blocking=True)
            e = e.cuda(self.local_rank, non_blocking=True)
            t = t.cuda(self.local_rank, non_blocking=True)
            with torch.amp.autocast(device_type='cuda', enabled=self.enable_amp):
                p = torch.sigmoid(self.model(x))
            preds.append(concat_all_gather(p))
            events.append(concat_all_gather(e))
            times.append(concat_all_gather(t))

        preds = torch.cat(preds).cpu().squeeze().numpy()   # risk scores (higher = riskier)
        events = torch.cat(events).cpu().squeeze().numpy() # binary labels (0/1)
        times  = torch.cat(times).cpu().squeeze().numpy()  # follow-up time (days)

        preds  = preds.reshape(-1)
        events = events.reshape(-1)
        times  = times.reshape(-1)

        # per-image predictions (inference only)
        if save_csv:
            self._save_sample_preds(
                self.prediction_loader.dataset.df, preds,
                self.prediction_loader.batch_size, 'prediction_preds.csv'
            )

        # Youden's J threshold (events vs preds)
        fpr, tpr, thr = roc_curve(events.astype(int), preds)
        youden = tpr - fpr
        best_idx = youden.argmax()
        best_thr = float(thr[best_idx])
        best_sens = float(tpr[best_idx])
        best_spec = float(1.0 - fpr[best_idx])

        # C-index (time-to-event)
        cindex = concordance_index(
            event_times=times,
            predicted_scores=-preds,     # higher risk -> shorter survival
            event_observed=events
        )

        if self.global_rank == 0:
            print(f"Prediction — C-index: {cindex:.3f}")
            print(f"Prediction — Youden's J best threshold: {best_thr:.6f} "
                  f"(sensitivity={best_sens:.3f}, specificity={best_spec:.3f})")

        self.best_threshold_prediction = best_thr
        self.best_sens_prediction = best_sens
        self.best_spec_prediction = best_spec
        return cindex

    # -------------------------
    # Early stopping + LR halving (model selection on validation AUROC)
    # -------------------------
    def train(self):
        best_auc = 0.
        stalls = 0

        for epoch in trange(self.args.epochs, desc='Epochs', disable=self.global_rank!=0):
            self.train_one_epoch(epoch)
            auc = self.valid_one_epoch(epoch, 'valid')
            improved = auc > (best_auc + 5e-4)
            if improved:
                best_auc = auc
                stalls = 0
                self.save()
            else:
                stalls += 1
                if stalls == self.args.lr_halve_patience:
                    for g in self.optimizer.param_groups:
                        g["lr"] *= 0.5
                    if self.global_rank == 0:
                        print("  ↳ LR halved")
                if stalls >= self.args.early_stop_patience:
                    if self.global_rank == 0:
                        print("  ↳ Early stop.")
                    break

    def inference(self):
        self.load()
        valid_auc = self.valid_one_epoch(self.args.epochs, 'valid', save_csv=True)
        test_auc = self.valid_one_epoch(self.args.epochs, 'test', save_csv=True)
        cindex = self.prediction(save_csv=True)

        if self.global_rank == 0:
            nan = float('nan')
            pd.DataFrame({
                'valid_auc': [valid_auc],
                'test_auc': [test_auc],
                'cindex': [cindex],
                'best_threshold_valid': [getattr(self, 'best_threshold_valid', nan)],
                'best_sensitivity_valid': [getattr(self, 'best_sens_valid', nan)],
                'best_specificity_valid': [getattr(self, 'best_spec_valid', nan)],
                'best_threshold_prediction': [getattr(self, 'best_threshold_prediction', nan)],
                'best_sensitivity_prediction': [getattr(self, 'best_sens_prediction', nan)],
                'best_specificity_prediction': [getattr(self, 'best_spec_prediction', nan)],
            }).to_csv(self.log_dir/'res.csv', index=False)
        dist.barrier()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dementia Detection Trainer')
    parser.add_argument('--arch', type=str,
                        choices=['retfound', 'mae', 'openclip', 'dinov2', 'dinov3', 'dinov3_vith16plus', 'retfound_dinov2'])
    parser.add_argument('--ft', type=str,
                        choices=['linear', 'partial', 'lora'])
    parser.add_argument('--lora_rank', type=int, default=4)
    parser.add_argument('--ft_blks', type=str, default='1',
                        help='number of final blocks to adapt, or "full" (all blocks)')
    # ---- paths (update to your environment) ----
    parser.add_argument('--det_csv', type=str, default='/path/to/dementia_detection.csv')
    parser.add_argument('--pred_csv', type=str, default='/path/to/dementia_prediction.csv')
    parser.add_argument('--root_dir', type=str, default='./ckpts')
    # ---- training ----
    parser.add_argument('--img_size', type=int, default=448)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=8, help='batch size per GPU')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--enable_amp', action='store_true')
    parser.add_argument('--lr_halve_patience', type=int, default=5)
    parser.add_argument('--early_stop_patience', type=int, default=10)
    parser.add_argument('--eval_only', action='store_true',
                        help='skip training; run inference with <root_dir>/<desc>/ckpt.pth.tar')

    args = parser.parse_args()

    # numeric strings -> int; otherwise lower-case string ("full")
    if isinstance(args.ft_blks, str) and args.ft_blks.isdigit():
        args.ft_blks = int(args.ft_blks)
    elif isinstance(args.ft_blks, str):
        args.ft_blks = args.ft_blks.lower()

    trainer = Trainer(args)
    if not args.eval_only:
        trainer.train()
    trainer.inference()
