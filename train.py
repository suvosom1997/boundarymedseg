# ==================== train.py ====================

import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

from model import BoundaryMedSeg
from configs import get_config
from losses import CombinedLoss
from datasets import get_dataset
from utils import set_seed, save_checkpoint, load_checkpoint

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    seg_loss_total = 0.0
    boundary_loss_total = 0.0
    
    pbar = tqdm(dataloader, desc='Training')
    for batch_idx, (images, masks) in enumerate(pbar):
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        seg_pred, boundary_pred = model(images)
        
        # Compute loss
        total_loss_batch, seg_loss_batch, boundary_loss_batch = criterion(seg_pred, boundary_pred, masks)
        
        # Backward pass
        total_loss_batch.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += total_loss_batch.item()
        seg_loss_total += seg_loss_batch.item()
        boundary_loss_total += boundary_loss_batch.item()
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{total_loss_batch.item():.4f}',
            'Seg': f'{seg_loss_batch.item():.4f}',
            'Boundary': f'{boundary_loss_batch.item():.4f}'
        })
    
    return total_loss / len(dataloader), seg_loss_total / len(dataloader), boundary_loss_total / len(dataloader)

def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0
    seg_loss_total = 0.0
    boundary_loss_total = 0.0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)
            
            # Forward pass
            seg_pred, boundary_pred = model(images)
            
            # Compute loss
            total_loss_batch, seg_loss_batch, boundary_loss_batch = criterion(seg_pred, boundary_pred, masks)
            
            total_loss += total_loss_batch.item()
            seg_loss_total += seg_loss_batch.item()
            boundary_loss_total += boundary_loss_batch.item()
            
            pbar.set_postfix({'Val Loss': f'{total_loss_batch.item():.4f}'})
    
    return total_loss / len(dataloader), seg_loss_total / len(dataloader), boundary_loss_total / len(dataloader)

def main():
    parser = argparse.ArgumentParser(description='Train BoundaryMedSeg')
    parser.add_argument('--dataset', required=True, 
                       choices=['busi', 'isic2018', 'brats2020', 'cvc', 'kvasir'],
                       help='Dataset name')
    parser.add_argument('--data_root', required=True, help='Root directory of dataset')
    parser.add_argument('--checkpoint_dir', default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--resume', default=None, help='Resume from checkpoint')
    parser.add_argument('--device', default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Get configuration
    config = get_config(args.dataset)
    print(f"Training BoundaryMedSeg on {config.dataset_name}")
    print(f"Input: {config.input_channels} channels, Output: {config.num_classes} classes")
    
    # Create model
    model = BoundaryMedSeg(config).to(args.device)
    
    # Create loss function
    criterion = CombinedLoss(seg_weight=1.0, boundary_weight=0.1)
    
    # Create optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # Create datasets
    train_dataset, val_dataset = get_dataset(args.dataset, args.data_root, config)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        print(f"Resumed from epoch {start_epoch}")
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Training loop
    print(f"Starting training for {config.num_epochs} epochs...")
    for epoch in range(start_epoch, config.num_epochs):
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        
        # Train
        train_loss, train_seg_loss, train_boundary_loss = train_epoch(
            model, train_loader, criterion, optimizer, args.device
        )
        
        # Validate
        val_loss, val_seg_loss, val_boundary_loss = validate_epoch(
            model, val_loader, criterion, args.device
        )
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Print epoch results
        print(f"Train Loss: {train_loss:.4f} (Seg: {train_seg_loss:.4f}, Boundary: {train_boundary_loss:.4f})")
        print(f"Val Loss: {val_loss:.4f} (Seg: {val_seg_loss:.4f}, Boundary: {val_boundary_loss:.4f})")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            print(f"🎉 New best validation loss: {best_val_loss:.4f}")
        
        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,
            'config': config
        }, is_best, args.checkpoint_dir, f'{args.dataset}_epoch_{epoch+1}.pth')
    
    print(f"\nTraining completed! Best validation loss: {best_val_loss:.4f}")

if __name__ == '__main__':
    main()
