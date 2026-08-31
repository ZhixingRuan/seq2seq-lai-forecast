from collections import defaultdict
from csv import writer
from glob import glob
from typing import Optional
from pathlib import Path
from contextlib import ExitStack

import torch
import torch.optim as optim
from torch.amp.grad_scaler import GradScaler
from torch.amp import autocast
from torch.utils.tensorboard.writer import SummaryWriter

from torch.profiler import tensorboard_trace_handler
from torch.profiler import profile as torch_profile
from torch.profiler import schedule as profiler_schedule

from model import ConvLSTMSeq2SeqDOY, ConvLSTMSeq2SeqDOY2Stacked

MODEL_REGISTRY = {
    'ConvLSTMSeq2SeqDOY': ConvLSTMSeq2SeqDOY,
    'ConvLSTMSeq2SeqDOY2Stacked': ConvLSTMSeq2SeqDOY2Stacked,
}
from loss import masked_mse_loss
from config import ProfilerConfig
from log import get_logger
from tensorboard_summary import MaskedRMSETracker

logger = get_logger(__name__)


def check_gpu_mem():
    device = torch.device('cuda')
    allocated_memory = torch.cuda.memory_allocated(device) / (1024 ** 2)  # in MB
    reserved_memory = torch.cuda.memory_reserved(device) / (1024 ** 2)  # in MB
    logger.info(f'allocated_memory: {allocated_memory}')
    logger.info(f'reserved_memory: {reserved_memory}')


def profiler(tensorboard_dir: Path):
    profiler_config = ProfilerConfig(
        on_trace_ready=tensorboard_trace_handler(tensorboard_dir),
        schedule=profiler_schedule(wait=20, warmup=1, active=10, repeat=10),
    )
    profiler = torch_profile(**profiler_config.model_dump())
    logger.info('Profiler active')
    return profiler


def save_checkpoint(
        model, optimizer, scaler, scheduler, epoch, best_loss, epochs_no_improve, filepath
    ):
    torch.save({
        'epoch':              epoch,
        'model_state_dict':   model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict':  scaler.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_loss':          best_loss,
        'epochs_no_improve':  epochs_no_improve,
    }, filepath)


def find_latest_checkpoint(checkpoint_dir):
    checkpoints = glob(f'{checkpoint_dir}/checkpoint_*.pth')
    if not checkpoints:
        return None
    return max(checkpoints)  # sorts by filename, epoch_030 > epoch_029 etc.


def train_convLSTM(
    train_loader,
    val_loader,
    input_dim,
    hidden_dim,
    kernel_size,
    lr,
    num_epochs,
    model_name: str = 'ConvLSTMSeq2SeqDOY',
    dropout: float = 0.2,
    tensorboard_dir: Optional[Path] = None,
    checkpoint_dir: Optional[Path] = None,
    checkpoint_epochs: int = 5,
    profile=False,
    patience: Optional[int] = None,
    min_delta: float = 0.0,
    scheduler_patience: int = 7,
    resume: bool = True,               # auto-resume if checkpoint exists
    resume_checkpoint_dir: Optional[Path] = None,  # where to look for checkpoints to resume from
    accumulation_steps: int = 1,
):
    scaler = GradScaler('cuda')
    torch.cuda.empty_cache()
    device = torch.device('cuda')

    model_cls = MODEL_REGISTRY[model_name]
    model = model_cls(
        input_dim=input_dim, target_dim=1,
        hidden_dim=hidden_dim, kernel_size=kernel_size,
        dropout=dropout if 'Stacked' in model_name else 0.0,
    )
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=scheduler_patience, min_lr=1e-6
    )

    # --- Auto-resume ---
    start_epoch       = 0
    best_loss         = float('inf')
    epochs_no_improve = 0

    if resume and resume_checkpoint_dir is not None:
        latest = find_latest_checkpoint(resume_checkpoint_dir)
        if latest:
            logger.info(f'Resuming training from checkpoint: {latest}')
            ckpt = torch.load(latest, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scaler.load_state_dict(ckpt['scaler_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])  # ADD THIS
            start_epoch       = ckpt['epoch'] + 1
            best_loss         = ckpt['best_loss']
            epochs_no_improve = ckpt['epochs_no_improve']
            logger.info(
                f'Resumed at epoch {start_epoch}, '
                f'best_loss={best_loss:.4f}, '
                f'patience={epochs_no_improve}/{patience}'
            )
        else:
            logger.info('No checkpoint found, starting training from scratch')

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Total trainable parameters: {trainable_params}')

    writer = SummaryWriter(tensorboard_dir)
    lag_rmse = defaultdict(lambda: MaskedRMSETracker(device))

    for epoch in range(start_epoch, num_epochs):
        logger.info('~' * 10 + f'Epoch {epoch} training' + '~' * 10)
        model.train()
        train_loss_sum = 0
        val_loss_sum   = 0

        with ExitStack() as stack:
            prof = stack.enter_context(profiler(tensorboard_dir)) if profile else None

            for batch_idx, (train_x, train_y, doy_x, doy_y, flag_y) in enumerate(train_loader):
                if batch_idx % accumulation_steps == 0:
                    optimizer.zero_grad()
                with autocast('cuda'):
                    check_gpu_mem()
                    train_x = train_x.to(device)
                    train_y = train_y.to(device)
                    doy_x   = doy_x.to(device)
                    doy_y   = doy_y.to(device)
                    flag_y  = flag_y.to(device)
                    output  = model(train_x, doy_x, doy_y, train_y.size(1))
                    check_gpu_mem()
                    if torch.isnan(train_x).any():
                        logger.warning(f'Batch {batch_idx}: NaN in train_x')
                    if torch.isnan(output).any():
                        logger.warning(f'Batch {batch_idx}: NaN in model output (shape={output.shape})')
                    loss    = masked_mse_loss(output, train_y, flag_y, valid_flag=1)
                    train_loss_sum += loss.item()
                    lss = loss / accumulation_steps

                scaler.scale(lss).backward()
                is_last_batch = (batch_idx + 1) == len(train_loader)
                if (batch_idx + 1) % accumulation_steps == 0 or is_last_batch:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                logger.info(f'Batch {batch_idx} loss: {loss.item()}')
                if profile:
                    prof.step()

        train_loss_avg = train_loss_sum / len(train_loader)
        logger.info(f'Epoch {epoch} average training loss: {train_loss_avg}')

        model.eval()
        with torch.no_grad():
            for val_x, val_y, doy_x, doy_y, flag_y in val_loader:
                val_x  = val_x.to(device)
                val_y  = val_y.to(device)
                doy_x  = doy_x.to(device)
                doy_y  = doy_y.to(device)
                flag_y = flag_y.to(device)
                output = model(val_x, doy_x, doy_y, val_y.size(1))
                loss   = masked_mse_loss(output, val_y, flag_y, valid_flag=1)
                val_loss_sum += loss.item()
                for lag in range(output.size(1)):
                    lag_rmse[lag].update(
                        output[:, lag, :, :, :],
                        val_y[:, lag, :, :, :],
                        flag_y[:, lag, :, :].unsqueeze(1),
                    )

            val_loss_avg = val_loss_sum / len(val_loader)
            logger.info(f'Epoch {epoch} average validation loss: {val_loss_avg}')

            scheduler.step(val_loss_avg)

            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('Learning Rate', current_lr, epoch)
            logger.info(f'Epoch {epoch} learning rate: {current_lr:.2e}')

            if val_loss_sum < best_loss - min_delta:
                best_loss         = val_loss_sum
                epochs_no_improve = 0
                # Save best weights immediately — not a reference
                torch.save(
                    model.state_dict(),
                    checkpoint_dir / 'best_model.pth'
                )
                logger.info(f'New best model saved at epoch {epoch}, '
                           f'val_loss={val_loss_avg:.4f}')
            else:
                epochs_no_improve += 1

            writer.add_scalars(
                'Validation LAI RMSE by Lag',
                {f'{lag:02}': tracker.compute()['rmse']
                 for lag, tracker in lag_rmse.items()},
                epoch,
            )
            writer.add_scalars(
                'Training vs. Validation Loss',
                {'Training': train_loss_avg, 'Validation': val_loss_avg},
                epoch,
            )
            for lag in lag_rmse:
                lag_rmse[lag].reset()

        if patience is not None and epochs_no_improve >= patience:
            logger.info(f'Early stopping at epoch {epoch} '
                       f'(no improvement for {patience} epochs)')
            break

        if epoch % checkpoint_epochs == 0:
            save_checkpoint(
                model, optimizer, scaler, scheduler,
                epoch, best_loss, epochs_no_improve,
                checkpoint_dir / f'checkpoint_{epoch:04d}.pth'
            )

    # Return best model by loading saved weights
    best_path = checkpoint_dir / 'best_model.pth'
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
        logger.info('Loaded best model weights for return')
    return model