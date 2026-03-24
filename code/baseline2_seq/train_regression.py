from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from dataset_structured import StructuredDataset, structured_collate_fn
from dataset_event import EventDataset, event_collate_fn
from model_multimodal_regression import MultiModalEventRegressor, MultiModalStructuredRegressor


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_world_info() -> Tuple[int, int, int]:
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    return rank, world_size, local_rank


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_main_process() -> bool:
    return (not is_distributed()) or dist.get_rank() == 0


def ddp_print(msg: str) -> None:
    if is_main_process():
        print(msg)


def init_distributed() -> Tuple[int, int, int]:
    rank, world_size, local_rank = get_world_info()
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError('DDP 多卡训练需要 CUDA 可用。')
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
    return rank, world_size, local_rank


def cleanup_distributed() -> None:
    if is_distributed():
        dist.barrier()
        dist.destroy_process_group()


def to_torch_batch(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, np.ndarray):
            if np.issubdtype(v.dtype, np.floating):
                out[k] = torch.from_numpy(v).float().to(device)
            elif np.issubdtype(v.dtype, np.integer):
                out[k] = torch.from_numpy(v).long().to(device)
            else:
                out[k] = v
        else:
            out[k] = v
    return out


def build_dataset_and_collate(args):
    processed = Path(args.processed_dir)
    if args.data_format == 'structured':
        ds = StructuredDataset(str(processed / 'multimodal_4d_tensors.npz'))
        collate = structured_collate_fn
    else:
        ds = EventDataset(str(processed / 'multimodal_event_sequences.npz'), str(processed / 'multimodal_event_index.csv'))
        collate = event_collate_fn
    return ds, collate


def split_indices_kfold(n: int, k: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:  #k折交叉验证
    indices = np.arange(n)
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)

    fold_sizes = np.full(k, n // k, dtype=int)
    fold_sizes[: n % k] += 1

    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    cur = 0
    for fs in fold_sizes:
        val_idx = indices[cur:cur + fs]
        train_idx = np.concatenate([indices[:cur], indices[cur + fs:]], axis=0)
        folds.append((train_idx, val_idx))
        cur += fs
    return folds


def split_indices_holdout(n: int, train_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    indices = np.arange(n)
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)
    n_train = max(1, int(n * train_ratio))
    train_idx = indices[:n_train]
    val_idx = indices[n_train:] if n_train < n else indices[-1:]
    return train_idx, val_idx


def build_loaders_for_indices(
    ds,
    collate,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    batch_size: int,
    num_workers: int,
    world_size: int,
    rank: int,
) -> Tuple[DataLoader, DataLoader]:
    train_subset = Subset(ds, train_idx.tolist())
    val_subset = Subset(ds, val_idx.tolist())

    train_sampler = None
    val_sampler = None
    if world_size > 1:
        train_sampler = DistributedSampler(train_subset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
        val_sampler = DistributedSampler(val_subset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate,
    )
    return train_loader, val_loader


def build_model(args, batch_example: Dict, device: torch.device) -> torch.nn.Module:
    rr_dim = int(batch_example['rr'].shape[-1])
    act_dim = int(batch_example['act'].shape[-1])
    sleep_dim = int(batch_example['sleep'].shape[-1])

    if args.data_format == 'structured':
        model = MultiModalStructuredRegressor(
            rr_dim,
            act_dim,
            sleep_dim,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            structured_max_hour=args.structured_max_hour,
            structured_cnn_layers=args.structured_cnn_layers,
            structured_cnn_kernel_size=args.structured_cnn_kernel_size,
            structured_use_hour_emb=args.structured_use_hour_emb,
            fusion_temporal_layers=args.fusion_temporal_layers,
        )
    else:
        model = MultiModalEventRegressor(
            rr_dim,
            act_dim,
            sleep_dim,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            event_time_feat_dim=args.event_time_feat_dim,
            event_n_heads=args.event_n_heads,
            event_n_layers=args.event_n_layers,
            event_ff_mult=args.event_ff_mult,
            event_dropout=args.event_dropout,
            event_use_time_feat=args.event_use_time_feat,
            event_max_tokens=args.event_max_tokens,
            fusion_temporal_layers=args.fusion_temporal_layers,
        )
    return model.to(device)


def masked_regression_loss(pred: torch.Tensor, y: torch.Tensor, day_mask: torch.Tensor):
    valid = (day_mask > 0.5) & torch.isfinite(y)
    if valid.sum() == 0:
        zero = torch.tensor(0.0, device=pred.device, requires_grad=True)
        return zero, torch.tensor(0.0, device=pred.device)

    diff = pred[valid] - y[valid]
    mse = (diff ** 2).mean()
    mae = diff.abs().mean()
    return mse, mae


def run_epoch(model, loader, optimizer, device, target_name: str, train: bool = True, fold_id: int = -1, epoch: int = -1):
    model.train() if train else model.eval()

    total_loss, total_mae, n_steps = 0.0, 0.0, 0
    for batch_idx, batch_np in enumerate(loader, start=1):
        batch = to_torch_batch(batch_np, device)
        y = batch[target_name]
        day_mask = batch['day_mask']

        # 仅在 event 模式下定位“某模态 T=0”的具体 batch / user  异常处理
        if 'rr_step_mask' in batch and 'act_step_mask' in batch and 'sleep_step_mask' in batch:
            rr_t = int(batch['rr'].shape[2])
            act_t = int(batch['act'].shape[2])
            sleep_t = int(batch['sleep'].shape[2])
            if rr_t == 0 or act_t == 0 or sleep_t == 0:
                mode = 'train' if train else 'val'
                user_ids = batch_np.get('user_id', []) if isinstance(batch_np, dict) else []
                ddp_print(
                    f'[DEBUG][FOLD {fold_id}][Epoch {epoch:03d}][{mode}][batch {batch_idx}] '
                    f'zero-T detected: rr_T={rr_t}, act_T={act_t}, sleep_T={sleep_t}, users={user_ids}'
                )

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            pred = model(batch)['pred']
            loss, mae = masked_regression_loss(pred, y, day_mask)
            if train:
                loss.backward()
                optimizer.step()

        total_loss += float(loss.detach().cpu())
        total_mae += float(mae.detach().cpu())
        n_steps += 1

    stats = torch.tensor([total_loss, total_mae, float(n_steps)], device=device, dtype=torch.float64)
    if is_distributed():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

    total_loss_all, total_mae_all, n_steps_all = stats.tolist()
    if n_steps_all <= 0:
        return 0.0, 0.0
    return total_loss_all / n_steps_all, total_mae_all / n_steps_all


def train_one_fold(args, fold_id: int, train_loader: DataLoader, val_loader: DataLoader, target_name: str, device: torch.device):
    first_batch = to_torch_batch(next(iter(train_loader)), device)
    model = build_model(args, first_batch, device)
    if is_distributed():
        # event分支在某些batch会出现某模态T=0，导致对应参数在该rank未参与反传。
        # DDP需要开启unused参数检测，避免跨rank梯度同步等待造成死锁。 主要改善在某些batch出现某模态T=0，导致对应参数在该rank未参与反传。
        model = DDP(model, device_ids=[device.index], output_device=device.index, find_unused_parameters=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau( #学习率调度器，这个是在验证指标出来之后调用，也就是在epoch验证结束后
        optimizer,
        mode='min',
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=args.lr_min,
    )
#     如果 va_loss 持续下降，学习率不变
# 如果 va_loss 很多轮都没有明显变好，就把学习率乘一个系数降下来

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / f'best_{args.data_format}_{target_name}_fold{fold_id}.pt'

    best_val = float('inf')
    best_val_mae = float('inf')
    no_improve_count = 0

    ddp_print(f'[FOLD {fold_id}] start')
    for epoch in range(1, args.epochs + 1):
        if hasattr(train_loader, 'sampler') and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        tr_loss, tr_mae = run_epoch(model, train_loader, optimizer, device, target_name, train=True, fold_id=fold_id, epoch=epoch)
        va_loss, va_mae = run_epoch(model, val_loader, optimizer, device, target_name, train=False, fold_id=fold_id, epoch=epoch)

        scheduler.step(va_loss)
        curr_lr = optimizer.param_groups[0]['lr']

        ddp_print(
            f'[FOLD {fold_id}][Epoch {epoch:03d}] '
            f'train_loss={tr_loss:.4f} train_mae={tr_mae:.4f} '
            f'val_loss={va_loss:.4f} val_mae={va_mae:.4f} lr={curr_lr:.2e}'
        )

        if va_loss < (best_val - args.early_stop_min_delta): #早停，连续30轮val loss没有改善就停止
            best_val = va_loss
            best_val_mae = va_mae
            no_improve_count = 0
            if is_main_process():
                model_state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
                torch.save({'model_state': model_state, 'args': vars(args), 'fold': fold_id}, ckpt_path)
                print(f'[FOLD {fold_id}] saved best -> {ckpt_path}')
        else:
            no_improve_count += 1
            if no_improve_count >= args.early_stop_patience:
                ddp_print(f'[FOLD {fold_id}] EARLY STOP at epoch {epoch}, best_val={best_val:.4f}')
                break

    return best_val, best_val_mae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_format', type=str, default='structured', choices=['structured', 'event'])
    parser.add_argument('--processed_dir', type=str, default='/home/wqshen/mm-test/code/baseline2_seq/processed')
    parser.add_argument(
        '--target',
        type=str,
        default='y_daily_stress',
        choices=['y_daily_stress', 'y_stai1', 'y_stai2', 'label_daily_stress', 'label_stai1', 'label_stai2'],
    )
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=128)

    # structured 日内建模可调参数
    parser.add_argument('--structured_max_hour', type=int, default=48)  #这个只是时间桶这个embedding表的大小，能最多支持48
    parser.add_argument('--structured_cnn_layers', type=int, default=2)
    parser.add_argument('--structured_cnn_kernel_size', type=int, default=3)
    parser.add_argument('--structured_use_hour_emb', action='store_true', default=True)
    parser.add_argument('--no_structured_use_hour_emb', action='store_false', dest='structured_use_hour_emb')

    # event 日内建模可调参数
    parser.add_argument('--event_time_feat_dim', type=int, default=2)
    parser.add_argument('--event_n_heads', type=int, default=4)
    parser.add_argument('--event_n_layers', type=int, default=2)
    parser.add_argument('--event_ff_mult', type=int, default=4)
    parser.add_argument('--event_dropout', type=float, default=0.1)
    parser.add_argument('--event_use_time_feat', action='store_true', default=True)
    parser.add_argument('--no_event_use_time_feat', action='store_false', dest='event_use_time_feat')
    parser.add_argument('--event_max_tokens', type=int, default=2048)  #event下每个模态进入transformer的最大token数量

    # 融合后的跨天序列建模（GRU）
    parser.add_argument('--fusion_temporal_layers', type=int, default=1)

    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='/home/wqshen/mm-test/code/baseline2_seq/checkpoints')

    parser.add_argument('--early_stop_patience', type=int, default=30)
    parser.add_argument('--early_stop_min_delta', type=float, default=1e-4)

    parser.add_argument('--lr_patience', type=int, default=10)
    parser.add_argument('--lr_factor', type=float, default=0.5)
    parser.add_argument('--lr_min', type=float, default=1e-6)

    parser.add_argument('--k_folds', type=int, default=6, help='=1时使用普通holdout；>1时执行K折交叉验证')
    parser.add_argument('--num_workers', type=int, default=0)

    args = parser.parse_args()

    rank, world_size, local_rank = init_distributed()
    set_seed(args.seed + rank)
    if world_size > 1:
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.data_format == 'structured' and args.target.startswith('label_'):
        args.target = args.target.replace('label_', 'y_')
    if args.data_format == 'event' and args.target.startswith('y_'):
        args.target = args.target.replace('y_', 'label_')

    ds, collate = build_dataset_and_collate(args)
    n = len(ds)
    target_name = args.target

    ddp_print(
        f'[INFO] rank={rank}/{world_size}, device={device}, data_format={args.data_format}, '
        f'target={target_name}, n_samples={n}'
    )

    if args.k_folds <= 1:
        train_idx, val_idx = split_indices_holdout(n, args.train_ratio, args.seed)
        train_loader, val_loader = build_loaders_for_indices(
            ds, collate, train_idx, val_idx, args.batch_size, args.num_workers, world_size, rank
        )
        best_val, best_val_mae = train_one_fold(args, 0, train_loader, val_loader, target_name, device)
        ddp_print(f'[FINAL] holdout best_val={best_val:.4f}, best_val_mae={best_val_mae:.4f}')
        cleanup_distributed()
        return

    k = int(args.k_folds)
    if k > n:
        k = n
        ddp_print(f'[WARN] k_folds 大于样本数，自动设为 k={k}')

    folds = split_indices_kfold(n, k, args.seed)
    fold_losses, fold_maes = [], []

    for fold_id, (train_idx, val_idx) in enumerate(folds, start=1):
        train_loader, val_loader = build_loaders_for_indices(
            ds, collate, train_idx, val_idx, args.batch_size, args.num_workers, world_size, rank
        )
        best_val, best_val_mae = train_one_fold(args, fold_id, train_loader, val_loader, target_name, device)
        fold_losses.append(best_val)
        fold_maes.append(best_val_mae)

    ddp_print('\n[K-FOLD SUMMARY]')
    for i, (l, m) in enumerate(zip(fold_losses, fold_maes), start=1):
        ddp_print(f'fold{i}: best_val={l:.4f}, best_val_mae={m:.4f}')
    ddp_print(f'mean best_val={np.mean(fold_losses):.4f} ± {np.std(fold_losses):.4f}')
    ddp_print(f'mean best_val_mae={np.mean(fold_maes):.4f} ± {np.std(fold_maes):.4f}')
    cleanup_distributed()


if __name__ == '__main__':
    main()
