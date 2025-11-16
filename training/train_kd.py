import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import time
import os

# Import our custom modules
from models.resnet import get_resnet18_teacher
from models.mobilenet import get_mobilenet_student
from utils.dataset import get_stl10_dataloaders
from utils.logger import Logger
from torchmetrics.classification import Accuracy # <-- New Import

class KDLoss(nn.Module):
    """
    Knowledge Distillation Loss Function
    As defined in your proposal: L_total = L_CE + alpha * L_KL
    
    """
    def __init__(self, alpha, temperature):
        super(KDLoss, self).__init__()
        self.alpha = alpha
        self.T = temperature
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_outputs, labels, teacher_outputs):
        # 1. Standard Cross-Entropy Loss (for hard labels)
        ce_loss = self.ce_loss(student_outputs, labels)
        
        # 2. KL Divergence Loss (for soft labels)
        # Soften both student and teacher outputs
        soft_student = F.log_softmax(student_outputs / self.T, dim=1)
        soft_teacher = F.softmax(teacher_outputs / self.T, dim=1)
        
        # Calculate KL divergence
        kl_loss = self.kl_loss(soft_student, soft_teacher) * (self.T ** 2)
        
        # [cite_start]3. Combined Loss [cite: 59]
        total_loss = (1.0 - self.alpha) * ce_loss + self.alpha * kl_loss
        
        return total_loss

def load_teacher(model_name, path, device):
    """Helper function to load a teacher model and its weights."""
    if model_name == 'resnet18':
        model = get_resnet18_teacher()
    else:
        raise ValueError(f"Unknown teacher model: {model_name}")
    
    try:
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        print(f"Loaded full teacher model from {path}")
    except RuntimeError:
        # This handles loading a backbone-only state_dict
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True), strict=False)
        print(f"Loaded teacher *backbone* from {path}. 'fc' layer is random.")
        
    model = model.to(device)
    model.eval() # Set teacher to evaluation mode
    return model

def get_teacher_logits(teachers, imgs):
    """
    Gets the logits from one or more teachers.
    If multiple teachers, this is an ENSEMBLE.
    """
    with torch.no_grad():
        all_logits = [teacher(imgs) for teacher in teachers]
        # Average the logits to get the ensemble prediction
        avg_logits = torch.stack(all_logits).mean(dim=0)
    return avg_logits

def train_one_epoch_kd(student, teachers, loader, kd_loss_fn, optimizer, device):
    """Runs a single KD training epoch."""
    student.train()
    total_loss = 0.0
    total_acc = 0.0
    
    for i, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)
        
        # Get teacher predictions (ensemble or single)
        teacher_logits = get_teacher_logits(teachers, imgs)
        
        # Forward pass for student
        student_logits = student(imgs)
        
        # Calculate KD loss
        loss = kd_loss_fn(student_logits, labels, teacher_logits)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        # Accuracy is still calculated on the hard labels
        total_acc += (student_logits.argmax(dim=1) == labels).float().mean().item()
        
    avg_loss = total_loss / len(loader)
    avg_acc = total_acc / len(loader)
    return avg_loss, avg_acc

def validate(model, loader, device, top5_acc_fn): # <-- Added top5_acc_fn
    """Validates the student model (standard validation)."""
    model.eval()
    total_acc = 0.0
    
    top5_acc_fn.reset() # <-- Reset metric
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(loader):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            total_acc += (outputs.argmax(dim=1) == labels).float().mean().item()
            top5_acc_fn.update(outputs, labels) # <-- Update Top-5
            
    avg_acc = total_acc / len(loader)
    avg_top5_acc = top5_acc_fn.compute().item() # <-- Compute Top-5
    return avg_acc, avg_top5_acc # <-- Return Top-5

def main():
    # --- 1. Argument Parsing ---
    parser = argparse.ArgumentParser(description="Knowledge Distillation Training")
    # Student
    parser.add_argument('--model', type=str, default='mobilenetv3', choices=['mobilenetv3'],
                        help="Student model")
    parser.add_argument('--student_load_path', type=str, default=None,
                        help="Path to pre-trained student (for progressive KD)")
    # Teachers
    parser.add_argument('--teachers', nargs='+', type=str, required=True,
                        help="List of teacher models (e.g., resnet18)")
    parser.add_argument('--teacher_paths', nargs='+', type=str, required=True,
                        help="List of paths to teacher checkpoints")
    # KD Params
    parser.add_argument('--alpha', type=float, default=0.5, help="Alpha for KD loss")
    parser.add_argument('--temp', type=float, default=4.0, help="Temperature for KD")
    # Training Params
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
    parser.add_argument('--save_name', type=str, required=True, help="Filename to save the model")
    
    args = parser.parse_args()

    if len(args.teachers) != len(args.teacher_paths):
        raise ValueError("Number of --teachers must match --teacher_paths")

    print(f"--- Starting Knowledge Distillation ---")
    
    # --- 2. Setup ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    os.makedirs('checkpoints', exist_ok=True)
    save_path = os.path.join('checkpoints', args.save_name)

    # --- 3. Load Data ---
    _, train_loader, test_loader = get_stl10_dataloaders(args.batch_size, ssl_transform=False)

    # --- 4. Load Models ---
    # Load Student
    print("Loading student model...")
    student = get_mobilenet_student().to(device)
    if args.student_load_path:
        student.load_state_dict(torch.load(args.student_load_path, map_location=device, weights_only=True))
        print(f"Loaded pre-trained student from {args.student_load_path}")
        
    # Load Teacher(s)
    print(f"Loading {len(args.teachers)} teacher model(s)...")
    teachers = [
        load_teacher(name, path, device) 
        for name, path in zip(args.teachers, args.teacher_paths)
    ]
    
    # --- 5. Define Loss, Optimizer, and Metrics ---
    # [cite_start]Temperature=4.0 as per your proposal [cite: 61]
    kd_loss_fn = KDLoss(alpha=args.alpha, temperature=args.temp)
    optimizer = optim.Adam(student.parameters(), lr=args.lr)
    top5_acc_fn = Accuracy(task="multiclass", num_classes=10, top_k=5).to(device) # <-- Init Top-5

    # --- 6. Initialize Logger ---
    log_filepath = os.path.join('logs', args.save_name.replace('.pth', '.csv'))
    logger = Logger(
        filepath=log_filepath,
        header=['epoch', 'train_loss', 'train_acc', 'test_acc', 'test_top5_acc', 'time'] # <-- Added Top-5
    )
    print(f"Logging results to {log_filepath}")

    # --- 7. Training Loop ---
    best_test_acc = 0.0
    start_time = time.time()

    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        
        train_loss, train_acc = train_one_epoch_kd(student, teachers, train_loader, kd_loss_fn, optimizer, device)
        test_acc, test_top5_acc = validate(student, test_loader, device, top5_acc_fn) # <-- Get Top-5
        
        epoch_duration = time.time() - epoch_start_time
        
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Test Acc: {test_acc:.4f} | "
              f"Test Top-5: {test_top5_acc:.4f} | " # <-- Added Top-5
              f"Time: {epoch_duration:.2f}s")
        
        # Log
        logger.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'test_top5_acc': test_top5_acc, # <-- Added Top-5
            'time': epoch_duration
        })
        
        # Save the best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(student.state_dict(), save_path)
            print(f"New best model saved to {save_path} (Test Acc: {best_test_acc:.4f})")
            
    total_time = time.time() - start_time
    print(f"\n--- Training Finished ---")
    print(f"Total time: {total_time:.2f}s")
    print(f"Best Test Accuracy: {best_test_acc:.4f}")
    print(f"Model saved to {save_path}")

if __name__ == '__main__':
    main()