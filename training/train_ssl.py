import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import time
import os
import copy
from tqdm import tqdm

# Import our custom modules
from models.resnet import get_simclr_model, get_byol_model, get_dino_model
from utils.dataset import get_stl10_dataloaders
from utils.logger import Logger

# =============================================================================
# === LOSS FUNCTIONS ==========================================================
# =============================================================================

class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (SimCLR)
    """
    def __init__(self, temperature, batch_size, device):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.batch_size = batch_size
        self.device = device
        self.mask = self.create_mask()
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def create_mask(self):
        mask = torch.ones(2 * self.batch_size, 2 * self.batch_size, dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(self.batch_size):
            mask[i, self.batch_size + i] = 0
            mask[self.batch_size + i, i] = 0
        return mask.to(self.device)

    def forward(self, z_i, z_j):
        z = torch.cat((z_i, z_j), dim=0)
        z = F.normalize(z, p=2, dim=1)
        sim_matrix = (z @ z.T) / self.temperature
        
        sim_i_j = torch.diag(sim_matrix, self.batch_size)
        sim_j_i = torch.diag(sim_matrix, -self.batch_size)
        
        positive_pairs = torch.cat((sim_i_j, sim_j_i), dim=0)
        negative_pairs = sim_matrix[self.mask].view(2 * self.batch_size, -1)
        
        labels = torch.zeros(2 * self.batch_size, dtype=torch.long, device=self.device)
        logits = torch.cat((positive_pairs.unsqueeze(1), negative_pairs), dim=1)
        
        loss = self.criterion(logits, labels)
        loss = loss / (2 * self.batch_size)
        return loss

class BYOLLoss(nn.Module):
    """
    Bootstrap Your Own Latent (BYOL) Loss
    Calculates the cosine similarity between the online network's predictions
    and the target network's projections.
    """
    def __init__(self):
        super(BYOLLoss, self).__init__()
        
    def normalized_mse(self, pred, target):
        pred_norm = F.normalize(pred, p=2, dim=1)
        target_norm = F.normalize(target, p=2, dim=1)
        # 2 - 2 * (pred_norm * target_norm).sum(dim=1)
        loss = 2.0 - 2.0 * (pred_norm * target_norm).sum(dim=1)
        return loss.mean()

    def forward(self, pred_view1, target_view2, pred_view2, target_view1):
        # Calculate loss for both views and average
        loss1 = self.normalized_mse(pred_view1, target_view2.detach())
        loss2 = self.normalized_mse(pred_view2, target_view1.detach())
        loss = (loss1 + loss2) / 2.0
        return loss

class DINOLoss(nn.Module):
    """
    Self-Distillation with No Labels (DINO) Loss
    - Student network is trained to match the output of the teacher network.
    - Teacher output is "sharpened" and "centered".
    """
    def __init__(self, out_dim, teacher_temp, student_temp, center_momentum=0.9):
        super(DINOLoss, self).__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        
        # Create a "center" buffer
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_output, teacher_output):
        """
        student_output: list of 2 tensors [ (B, D), (B, D) ]
        teacher_output: list of 2 tensors [ (B, D), (B, D) ]
        """
        student_out = student_output / self.student_temp
        
        # Teacher: Sharpening and Centering
        teacher_out = F.softmax((teacher_output - self.center) / self.teacher_temp, dim=-1)
        
        # Calculate cross-entropy loss
        # We want the student to predict the teacher's distribution
        loss = -torch.sum(teacher_out.detach() * F.log_softmax(student_out, dim=-1), dim=-1).mean()
        
        self.update_center(teacher_output)
        return loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
        # Exponential moving average update
        self.center = self.center * self.center_momentum + \
                      batch_center * (1 - self.center_momentum)

# =============================================================================
# === HELPER: MOMENTUM UPDATES ================================================
# =============================================================================

@torch.no_grad()
def momentum_update_target(online_model, target_model, m):
    """
    Performs an exponential moving average update on the target model.
    target_weights = m * target_weights + (1 - m) * online_weights
    """
    for param_ol, param_tgt in zip(online_model.parameters(), target_model.parameters()):
        param_tgt.data.mul_(m).add_(param_ol.data, alpha=1 - m)

# =============================================================================
# === EPOCH TRAINING FUNCTIONS ================================================
# =============================================================================

def train_one_epoch_simclr(model, loader, loss_fn, optimizer, device):
    total_loss = 0.0
    for (view1, view2), _ in tqdm(loader, desc="SimCLR Epoch", leave=False):
        view1, view2 = view1.to(device), view2.to(device)
        
        proj1 = model(view1)
        proj2 = model(view2)
        
        loss = loss_fn(proj1, proj2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(loader)

def train_one_epoch_byol(online_net, target_net, loader, loss_fn, optimizer, m, device):
    total_loss = 0.0
    for (view1, view2), _ in tqdm(loader, desc="BYOL Epoch", leave=False):
        view1, view2 = view1.to(device), view2.to(device)
        
        # Forward pass online network
        pred1 = online_net(view1)
        pred2 = online_net(view2)
        
        # Forward pass target network
        with torch.no_grad():
            target1 = target_net.get_projection(view1)
            target2 = target_net.get_projection(view2)
        
        loss = loss_fn(pred1, target2, pred2, target1)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update the target network
        momentum_update_target(online_net, target_net, m)
        
        total_loss += loss.item()
    return total_loss / len(loader)

def train_one_epoch_dino(student, teacher, loader, loss_fn, optimizer, m, device):
    total_loss = 0.0
    for (view1, view2), _ in tqdm(loader, desc="DINO Epoch", leave=False):
        view1, view2 = view1.to(device), view2.to(device)

        # Teacher forward pass
        with torch.no_grad():
            teacher_out1 = teacher(view1)
            teacher_out2 = teacher(view2)
            
        # Student forward pass
        student_out1 = student(view1)
        student_out2 = student(view2)
        
        # Calculate loss
        # Note: DINO's loss is asymmetric, it only matches view1_student -> view2_teacher
        # and view2_student -> view1_teacher
        loss = loss_fn(student_out1, teacher_out2) + \
               loss_fn(student_out2, teacher_out1)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update the teacher network
        momentum_update_target(student, teacher, m)
        
        total_loss += loss.item()
    return total_loss / len(loader)

# =============================================================================
# === MAIN FUNCTION ===========================================================
# =============================================================================

def main():
    # --- 1. Argument Parsing ---
    parser = argparse.ArgumentParser(description="SSL Pretraining (SimCLR, BYOL, DINO)")
    parser.add_argument('--method', type=str, required=True, choices=['simclr', 'byol', 'dino'],
                        help="SSL method to use")
    parser.add_argument('--epochs', type=int, default=100, help="Number of pretraining epochs")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size")
    parser.add_argument('--momentum', type=float, default=0.99, help="Momentum for target networks (BYOL/DINO)")
    parser.add_argument('--temp', type=float, default=0.5, help="Temperature (SimCLR/DINO)")
    parser.add_argument('--save_name', type=str, required=True, help="Filename to save the backbone")
    args = parser.parse_args()

    print(f"--- Starting {args.method.upper()} Pretraining ---")
    
    # --- 2. Setup ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    os.makedirs('checkpoints', exist_ok=True)
    save_path = os.path.join('checkpoints', args.save_name)

    # --- 3. Load Data ---
    unlabeled_loader, _, _ = get_stl10_dataloaders(args.batch_size, ssl_transform=True)

    # --- 4. Load Model, Loss, and Optimizer ---
    if args.method == 'simclr':
        model = get_simclr_model().to(device)
        loss_fn = NTXentLoss(temperature=args.temp, batch_size=args.batch_size, device=device).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
        
    elif args.method == 'byol':
        online_net = get_byol_model().to(device)
        target_net = copy.deepcopy(online_net).to(device) # Create target network
        loss_fn = BYOLLoss().to(device)
        optimizer = optim.Adam(online_net.parameters(), lr=args.lr, weight_decay=1e-6)
        
    elif args.method == 'dino':
        student = get_dino_model().to(device)
        teacher = copy.deepcopy(student).to(device) # Create teacher network
        loss_fn = DINOLoss(out_dim=4096, teacher_temp=0.04, student_temp=args.temp).to(device)
        optimizer = optim.Adam(student.parameters(), lr=args.lr, weight_decay=1e-6)

    # --- 5. Initialize Logger ---
    log_filepath = os.path.join('logs', args.save_name.replace('.pth', '.csv'))
    logger = Logger(
        filepath=log_filepath,
        header=['epoch', 'ssl_loss', 'time']
    )
    print(f"Logging results to {log_filepath}")

    # --- 6. Training Loop ---
    start_time = time.time()
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        
        # --- Call the correct training function ---
        if args.method == 'simclr':
            train_loss = train_one_epoch_simclr(model, unlabeled_loader, loss_fn, optimizer, device)
            model_to_save = model # SimCLR has one model
            
        elif args.method == 'byol':
            train_loss = train_one_epoch_byol(
                online_net, target_net, unlabeled_loader, loss_fn, optimizer, args.momentum, device
            )
            model_to_save = online_net # Save the online network
            
        elif args.method == 'dino':
            train_loss = train_one_epoch_dino(
                student, teacher, unlabeled_loader, loss_fn, optimizer, args.momentum, device
            )
            model_to_save = student # Save the student network
        
        epoch_duration = time.time() - epoch_start_time
        
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"SSL Loss: {train_loss:.4f} | "
              f"Time: {epoch_duration:.2f}s")
        
        logger.log({ 'epoch': epoch + 1, 'ssl_loss': train_loss, 'time': epoch_duration })
        
    # --- 7. Save the Backbone ---
    # We only save the backbone (ResNet-18)
    
    # --- THIS IS THE FIX ---
    torch.save(model_to_save.backbone.state_dict(), save_path)
    # ----------------------
    
    total_time = time.time() - start_time
    print(f"\n--- Pretraining Finished ---")
    print(f"Total time: {total_time:.2f}s")
    print(f"Backbone model saved to {save_path}")

if __name__ == '__main__':
    main()