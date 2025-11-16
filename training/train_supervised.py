import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import time
import os

# Import our custom modules
from models.resnet import get_resnet18_teacher
from models.mobilenet import get_mobilenet_student
from utils.dataset import get_stl10_dataloaders
from utils.logger import Logger
from torchmetrics.classification import Accuracy # <-- New Import

# Define a simple accuracy function (for Top-1)
def calculate_accuracy(y_pred, y_true):
    return (y_pred.argmax(dim=1) == y_true).float().mean()

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Runs a single training epoch."""
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    
    for i, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_acc += calculate_accuracy(outputs, labels).item()
        
    avg_loss = total_loss / len(loader)
    avg_acc = total_acc / len(loader)
    return avg_loss, avg_acc

def validate(model, loader, criterion, device, top5_acc_fn): # <-- Added top5_acc_fn
    """Validates the model on the test/validation set."""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    
    top5_acc_fn.reset() # <-- Reset metric
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(loader):
            imgs, labels = imgs.to(device), labels.to(device)
            
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            total_acc += calculate_accuracy(outputs, labels).item()
            top5_acc_fn.update(outputs, labels) # <-- Update Top-5
            
    avg_loss = total_loss / len(loader)
    avg_acc = total_acc / len(loader)
    avg_top5_acc = top5_acc_fn.compute().item() # <-- Compute Top-5
    
    return avg_loss, avg_acc, avg_top5_acc # <-- Return Top-5

def main():
    # --- 1. Argument Parsing ---
    parser = argparse.ArgumentParser(description="Supervised Baseline Training")
    parser.add_argument('--model', type=str, required=True, choices=['resnet18', 'mobilenetv3'],
                        help="Model to train (resnet18 or mobilenetv3)")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
    parser.add_argument('--save_name', type=str, required=True, help="Filename to save the model")
    parser.add_argument('--load_path', type=str, default=None,
                        help="Path to pre-trained backbone (for fine-tuning)")
    args = parser.parse_args()

    print(f"--- Starting Supervised Training for {args.model} ---")
    
    # --- 2. Setup ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create checkpoints directory
    os.makedirs('checkpoints', exist_ok=True)
    save_path = os.path.join('checkpoints', args.save_name)
    
    # --- 3. Load Data ---
    # We don't need the unlabeled loader here
    _, train_loader, test_loader = get_stl10_dataloaders(args.batch_size, ssl_transform=False)

    # --- 4. Load Model ---
    if args.model == 'resnet18':
        model = get_resnet18_teacher()
    elif args.model == 'mobilenetv3':
        model = get_mobilenet_student()
    
    model = model.to(device)
    
    if args.load_path:
        # Load the backbone weights from an SSL-pretrained model
        # We use strict=False because the backbone file does not
        # contain weights for the final 'fc' (classifier) layer.
        try:
            model.load_state_dict(torch.load(args.load_path, map_location=device, weights_only=True), strict=False)
            print(f"Loaded SSL backbone weights from {args.load_path}")
        except Exception as e:
            print(f"Error loading backbone: {e}. Training from scratch.")

    # --- 5. Define Loss, Optimizer, and Metrics ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    top5_acc_fn = Accuracy(task="multiclass", num_classes=10, top_k=5).to(device) # <-- Init Top-5

    # --- 6. Initialize Logger ---
    log_filepath = os.path.join('logs', args.save_name.replace('.pth', '.csv'))
    logger = Logger(
        filepath=log_filepath,
        header=['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc', 'test_top5_acc', 'time'] # <-- Added Top-5
    )
    print(f"Logging results to {log_filepath}")

    # --- 7. Training Loop ---
    best_test_acc = 0.0
    start_time = time.time()

    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc, test_top5_acc = validate(model, test_loader, criterion, device, top5_acc_fn) # <-- Get Top-5
        
        epoch_duration = time.time() - epoch_start_time
        
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | "
              f"Test Top-5: {test_top5_acc:.4f} | " # <-- Added Top-5
              f"Time: {epoch_duration:.2f}s")
        
        # Log the data
        logger.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'test_top5_acc': test_top5_acc, # <-- Added Top-5
            'time': epoch_duration
        })
        
        # Save the best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved to {save_path} (Test Acc: {best_test_acc:.4f})")
            
    total_time = time.time() - start_time
    print(f"\n--- Training Finished ---")
    print(f"Total time: {total_time:.2f}s")
    print(f"Best Test Accuracy: {best_test_acc:.4f}")
    print(f"Model saved to {save_path}")

if __name__ == '__main__':
    main()