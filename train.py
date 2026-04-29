"""
train.py - ONN Training Script
==============================

Training loop with checkpointing, early stopping, and visualization.
Hardware params from config.py, training-specific params below.

Usage: python train.py
"""

import torch
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau, LinearLR, SequentialLR
from pathlib import Path
import time
import csv

from config import Config
from data import get_mnist_loaders
from model import OpticalNeuralNetwork


# =============================================================================
# TRAINING PARAMETERS
# =============================================================================
#
# SELECT WHICH RUN TO EXECUTE — change RUN to one of:
#
#   'binary_3x3'    Binary encoding,   3×3 zones  → checkpoints_binary_3x3
#   'center_3x3'    Centre-based,      3×3 zones  → checkpoints_center_3x3
#   'maxzone_4x4'   Max-zone,          4×4 zones  → checkpoints_maxzone_4x4
#   'maxzone_5x5'   Max-zone,          5×5 zones  → checkpoints_maxzone_5x5  (REFERENCE — already 90.85%)
#   'zone'          Direct zone,       2×5 zones  → checkpoints_zone
#
# All runs use identical hyperparameters (AdamW, plateau scheduler, warmup,
# gradient clip, label smoothing) so results are directly comparable.
# =============================================================================

RUN = 'binary_3x3'   # <-- CHANGE THIS LINE ONLY

_PRESETS = {
    'binary_3x3':  dict(method='binary',   rows=3, cols=3, save='checkpoints_binary_3x3'),
    'center_3x3':  dict(method='center',   rows=3, cols=3, save='checkpoints_center_3x3'),
    'maxzone_4x4': dict(method='maxzone',  rows=4, cols=4, save='checkpoints_maxzone_4x4'),
    'maxzone_5x5': dict(method='maxzone',  rows=5, cols=5, save='checkpoints_maxzone_5x5'),
    'zone':        dict(method='zone',     rows=2, cols=5, save='checkpoints_zone'),
}
assert RUN in _PRESETS, f"Unknown RUN='{RUN}'. Choose from: {list(_PRESETS)}"
_p = _PRESETS[RUN]

DETECTION_METHOD = _p['method']
DETECTOR_ROWS    = _p['rows']
DETECTOR_COLS    = _p['cols']
SAVE_DIR         = _p['save']

# --- Shared hyperparameters (identical for all runs) ---
EARLY_STOPPING   = True
PATIENCE         = 30     # Stop if no improvement for N epochs
PLATEAU_FACTOR   = 0.5    # LR *= factor when plateau detected
PLATEAU_PATIENCE = 10     # Epochs to wait before reducing LR
MIN_LR           = 1e-6   # Stop reducing LR below this
MAX_EPOCHS       = 500    # Safety cap (early stopping should fire first)
LABEL_SMOOTHING  = 0.1    # Label smoothing coefficient

RESUME = None             # None, 'latest', or path to a checkpoint file
DEVICE = None             # None=auto, 'cuda', or 'cpu'


# =============================================================================
# UTILITIES
# =============================================================================

class AverageMeter:
    """Running average tracker."""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.avg = self.sum = self.count = 0
    
    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target):
    """Compute top-1 accuracy (%)."""
    with torch.no_grad():
        pred = output.argmax(dim=1)
        return (pred == target).float().mean() * 100


# =============================================================================
# CHECKPOINTING
# =============================================================================

def save_checkpoint(model, optimizer, scheduler, epoch, best_acc, save_dir, name='best'):
    """Save best checkpoint (overwrites previous)."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_acc': best_acc,
    }, save_dir / f'{name}.pt')


def save_best_masks(model, save_dir, config):
    """Save masks as .pt and .png for LCD display."""
    from visualize_propagation import save_masks_as_images
    
    masks_dir = Path(save_dir) / 'best_masks'
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    masks = [layer.get_amplitude(
        binary=config.lcd.is_binary,
        min_transmission=config.lcd.min_transmission if config.lcd.min_transmission > 0 else None,
        max_transmission=config.lcd.max_transmission if config.lcd.max_transmission < 1 else None
    ).detach().cpu() for layer in model.layers]
    
    torch.save(masks, masks_dir / 'masks.pt')
    save_masks_as_images(masks, str(masks_dir))


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """Load checkpoint, return (epoch, best_acc)."""
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler and ckpt['scheduler_state_dict']:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    return ckpt['epoch'], ckpt['best_acc']


def find_latest_checkpoint(save_dir):
    """Find latest checkpoint in directory."""
    save_dir = Path(save_dir)
    if (save_dir / 'best.pt').exists():
        return save_dir / 'best.pt'
    checkpoints = list(save_dir.glob('checkpoint_epoch*.pt'))
    if checkpoints:
        checkpoints.sort(key=lambda x: int(x.stem.split('epoch')[-1]))
        return checkpoints[-1]
    return None


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_training(model, save_dir, epoch, images=None, labels=None):
    """Save mask and prediction visualizations."""
    import matplotlib.pyplot as plt
    
    vis_dir = Path(save_dir) / 'visualizations'
    vis_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    
    # Masks
    num_layers = len(model.layers)
    fig, axes = plt.subplots(1, num_layers, figsize=(4*num_layers, 4))
    if num_layers == 1:
        axes = [axes]
    
    for i, layer in enumerate(model.layers):
        amp = layer.get_amplitude(
            binary=model.config.lcd.is_binary,
            min_transmission=model.config.lcd.min_transmission,
            max_transmission=model.config.lcd.max_transmission
        ).detach().cpu().numpy()
        axes[i].imshow(amp, cmap='gray', vmin=0, vmax=1)
        axes[i].set_title(f'Layer {i+1}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(vis_dir / f'masks_epoch{epoch:03d}.png', dpi=150)
    plt.close()
    
    # Predictions
    if images is not None and labels is not None:
        with torch.no_grad():
            preds = model(images[:16]).argmax(dim=1)
        
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        for i, ax in enumerate(axes.flat):
            if i < len(images):
                ax.imshow(images[i, 0].cpu().numpy(), cmap='gray')
                color = 'green' if preds[i] == labels[i] else 'red'
                ax.set_title(f'P:{preds[i].item()} T:{labels[i].item()}', color=color)
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(vis_dir / f'predictions_epoch{epoch:03d}.png', dpi=150)
        plt.close()


# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(model, loader, optimizer, device, scaler=None):
    """Train one epoch. Returns (loss, accuracy)."""
    model.train()
    losses, accs = AverageMeter(), AverageMeter()
    use_amp = scaler is not None
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        with torch.amp.autocast('cuda', enabled=use_amp):
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels, label_smoothing=LABEL_SMOOTHING)
        
        optimizer.zero_grad()
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        losses.update(loss.item(), images.size(0))
        accs.update(accuracy(outputs, labels).item(), images.size(0))
    
    return losses.avg, accs.avg


@torch.no_grad()
def validate(model, loader, device):
    """Validate model. Returns (loss, accuracy)."""
    model.eval()
    losses, accs = AverageMeter(), AverageMeter()
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        
        losses.update(loss.item(), images.size(0))
        accs.update(accuracy(outputs, labels).item(), images.size(0))
    
    return losses.avg, accs.avg


def train(config=None, detection_method='center', epochs=None, resume=None,
          save_dir='checkpoints', device=None, **detection_kwargs):
    """Main training function."""
    if config is None:
        config = Config()
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    num_epochs = MAX_EPOCHS
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Device: {device}, Method: {detection_method}, Epochs: {num_epochs}")
    
    # Data
    train_loader, test_loader = get_mnist_loaders(batch_size=config.training.batch_size)
    print(f"Train: {len(train_loader.dataset)}, Test: {len(test_loader.dataset)}")
    
    # Model
    model = OpticalNeuralNetwork(config, detection_method, **detection_kwargs).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer with per-component learning rates
    mask_params = [p for layer in model.layers for p in layer.parameters()]
    detector_params = [p for p in model.detector.parameters()]
    
    mask_lr = config.training.amplitude_lr
    detector_lr = config.training.detection_lr
    wd = config.training.weight_decay
    
    param_groups = [
        {'params': mask_params,     'lr': mask_lr,     'weight_decay': 0.0},     # No decay on masks (physics-constrained via sigmoid)
        {'params': detector_params, 'lr': detector_lr, 'weight_decay': wd},
    ]
    print(f"  Mask LR: {mask_lr}, Detector LR: {detector_lr}, Weight decay: {wd}")
    
    optimizer = AdamW(param_groups)
    
    warmup_epochs = config.training.warmup_epochs
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    if use_amp:
        print(f"  Mixed precision (AMP) enabled")
    
    # Plateau scheduler: LR stays high while model improves, drops when it stalls
    plateau = ReduceLROnPlateau(optimizer, mode='max', factor=PLATEAU_FACTOR,
                                patience=PLATEAU_PATIENCE, min_lr=MIN_LR)
    
    if warmup_epochs > 0:
        warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
        in_warmup = True
        print(f"  Warmup: {warmup_epochs} epochs, then plateau (factor={PLATEAU_FACTOR}, patience={PLATEAU_PATIENCE})")
    else:
        in_warmup = False
        print(f"  Plateau scheduler (factor={PLATEAU_FACTOR}, patience={PLATEAU_PATIENCE})")
    
    # Resume
    start_epoch, best_acc = 0, 0.0
    if resume:
        ckpt_path = find_latest_checkpoint(save_dir) if resume == 'latest' else Path(resume)
        if ckpt_path and ckpt_path.exists():
            print(f"Resuming from {ckpt_path}")
            start_epoch, best_acc = load_checkpoint(model, optimizer, None, ckpt_path)
            start_epoch += 1
            if start_epoch >= warmup_epochs:
                in_warmup = False
    
    # Training loop
    epochs_no_improve = 0
    
    # CSV training log
    csv_path = save_dir / 'training_log.csv'
    csv_existed = csv_path.exists() and start_epoch > 0
    csv_file = open(csv_path, 'a' if csv_existed else 'w', newline='')
    csv_writer = csv.writer(csv_file)
    if not csv_existed:
        csv_writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc',
                             'mask_lr', 'detector_lr', 'temperature', 'logit_scale', 'best_acc'])
        csv_file.flush()
    
    print(f"\n{'='*60}\nStarting training...\n{'='*60}\n")
    
    for epoch in range(start_epoch, num_epochs):
        t0 = time.time()
        
        # Temperature tied to LR: τ = 0.5 + 1.5 × (current_lr / initial_lr)
        # High LR → high τ (explore), Low LR → low τ (commit)
        if hasattr(model.detector, 'temperature'):
            lr_ratio = optimizer.param_groups[0]['lr'] / mask_lr
            model.detector.temperature = 0.5 + 1.5 * lr_ratio
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, scaler)
        val_loss, val_acc = validate(model, test_loader, device)
        
        # Step scheduler: warmup first, then plateau
        if in_warmup and epoch < warmup_epochs:
            warmup.step()
        else:
            in_warmup = False
            plateau.step(val_acc)
        mask_lr_now = optimizer.param_groups[0]['lr']
        det_lr_now = optimizer.param_groups[1]['lr']
        temp_now = model.detector.temperature if hasattr(model.detector, 'temperature') else None
        scale_now = model.detector.logit_scale.item() if hasattr(model.detector, 'logit_scale') else None
        
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train: {train_loss:.4f}/{train_acc:.1f}% | "
              f"Val: {val_loss:.4f}/{val_acc:.1f}% | "
              f"mLR: {mask_lr_now:.1e} dLR: {det_lr_now:.1e}"
              f"{f' | τ={temp_now:.2f}' if temp_now else ''}"
              f"{f' | s={scale_now:.1f}' if scale_now else ''}"
              f" | {time.time()-t0:.1f}s")
        
        # Write CSV row
        csv_writer.writerow([epoch+1, f'{train_loss:.6f}', f'{train_acc:.2f}',
                             f'{val_loss:.6f}', f'{val_acc:.2f}',
                             f'{mask_lr_now:.2e}', f'{det_lr_now:.2e}',
                             f'{temp_now:.4f}' if temp_now else '',
                             f'{scale_now:.4f}' if scale_now else '',
                             f'{best_acc:.2f}'])
        csv_file.flush()
        
        # Best model
        if val_acc > best_acc:
            best_acc = val_acc
            epochs_no_improve = 0
            save_checkpoint(model, optimizer, None, epoch, best_acc, save_dir)
            save_best_masks(model, save_dir, config)
            print(f"  ★ New best: {best_acc:.2f}%")
        else:
            epochs_no_improve += 1
            if EARLY_STOPPING and epochs_no_improve >= PATIENCE:
                print(f"\n⚠ Early stopping after {PATIENCE} epochs without improvement")
                break
        
        # Visualize every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            vis_images, vis_labels = next(iter(test_loader))
            visualize_training(model, save_dir, epoch, vis_images.to(device), vis_labels.to(device))
    
    csv_file.close()
    print(f"Training log saved to: {csv_path}")
    print(f"\n{'='*60}\nTraining complete! Best: {best_acc:.2f}%\n{'='*60}")
    return model


# =============================================================================
# MAIN
# =============================================================================

def main():
    config = Config()
    device = torch.device(DEVICE) if DEVICE else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print("ONN TRAINING")
    print("=" * 60)
    print(f"\nConfig: {config.lcd.resolution[0]}×{config.lcd.resolution[0]}, "
          f"{config.multilayer.num_layers} layers, binary={config.lcd.is_binary}")
    print(f"Training: max {MAX_EPOCHS} epochs, batch={config.training.batch_size}, "
          f"lr={config.training.learning_rate}")
    print(f"Detection: {DETECTION_METHOD}" + 
          (f" ({DETECTOR_ROWS}×{DETECTOR_COLS})" if DETECTION_METHOD != 'zone' else " (2×5)"))
    print(f"Scheduler: plateau (patience={PLATEAU_PATIENCE}, factor={PLATEAU_FACTOR}), "
          f"Early stop: {PATIENCE} epochs, Device: {device}, Save: {SAVE_DIR}\n")
    
    return train(
        config=config,
        detection_method=DETECTION_METHOD,
        epochs=config.training.epochs,
        resume=RESUME,
        save_dir=SAVE_DIR,
        device=device,
        rows=DETECTOR_ROWS,
        cols=DETECTOR_COLS
    )


if __name__ == '__main__':
    main()