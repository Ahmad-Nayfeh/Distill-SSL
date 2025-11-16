import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# --- Supervised Teacher Model ---

def get_resnet18_teacher():
    """
    Returns a ResNet-18 model, which will serve as our standard
    supervised teacher (T-SUP).
    """
    model = models.resnet18(weights=None) 
    
    # STL-10 has 10 classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    
    return model

# --- SSL Model Components ---

class MLPHead(nn.Module):
    """
    Helper class to build a multi-layer perceptron (MLP) head.
    This is used for the projection/prediction heads in SimCLR, BYOL, etc.
    
    Architecture:
    (Linear -> BatchNorm -> ReLU) * (num_layers - 1) -> Linear
    """
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2):
        super(MLPHead, self).__init__()
        layers = []
        
        # Input layer
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            
        # Output layer
        layers.append(nn.Linear(hidden_dim, out_dim))
        
        self.head = nn.Sequential(*layers)

    def forward(self, x):
        return self.head(x)

class DINOHead(nn.Module):
    """
    Specific projection head for DINO.
    - 3-layer MLP
    - Hidden layers use BatchNorm
    - Output layer uses WeightNorm
    - Includes a "bottleneck" to reduce dimension before the final layer
    """
    def __init__(self, in_dim, hidden_dim, bottleneck_dim, out_dim):
        super().__init__()
        # Input layer
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        # Hidden layer
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        # Bottleneck layer
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        
        # Output layer
        self.last_layer = nn.utils.parametrizations.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        
        # --- THIS IS THE FIX ---
        # Access the weight via .original0 instead of .original
        self.last_layer.parametrizations.weight.original0.data.fill_(1)
        # ---------------------

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.normalize(x, dim=-1, p=2) # L2 normalize
        x = self.last_layer(x)
        return x

# --- Main SSL Model Wrappers ---

class SimCLR(nn.Module):
    """
    Wrapper for SimCLR.
    ResNet-18 Backbone + 2-Layer MLP Projection Head
    """
    def __init__(self, backbone, proj_dim=128):
        super().__init__()
        self.backbone = backbone
        num_ftrs = self.backbone.fc.in_features
        
        # Replace original classifier with identity
        self.backbone.fc = nn.Identity() 
        
        # Add projection head
        self.projection_head = MLPHead(
            in_dim=num_ftrs,
            hidden_dim=num_ftrs, # Use 512 as hidden dim
            out_dim=proj_dim,
            num_layers=2
        )
        
    def forward(self, x):
        features = self.backbone(x)
        projections = self.projection_head(features)
        return projections

class BYOL(nn.Module):
    """
    Wrapper for BYOL.
    ResNet-18 Backbone + Projection Head + Prediction Head
    """
    def __init__(self, backbone, proj_hidden_dim=2048, proj_dim=128, pred_hidden_dim=512):
        super().__init__()
        self.backbone = backbone
        num_ftrs = self.backbone.fc.in_features # 512
        
        # Replace original classifier with identity
        self.backbone.fc = nn.Identity()
        
        # Add projection head (e.g., 512 -> 2048 -> 128)
        self.projection_head = MLPHead(
            in_dim=num_ftrs,
            hidden_dim=proj_hidden_dim,
            out_dim=proj_dim,
            num_layers=2
        )
        
        # Add prediction head (e.g., 128 -> 512 -> 128)
        self.prediction_head = MLPHead(
            in_dim=proj_dim,
            hidden_dim=pred_hidden_dim,
            out_dim=proj_dim,
            num_layers=2
        )

    def forward(self, x):
        # This forward pass is for the "online" network
        features = self.backbone(x)
        projections = self.projection_head(features)
        predictions = self.prediction_head(projections)
        return predictions

    def get_projection(self, x):
        # Helper to get projections (for the "target" network)
        features = self.backbone(x)
        projections = self.projection_head(features)
        return projections

class DINO(nn.Module):
    """
    Wrapper for DINO.
    ResNet-18 Backbone + DINOHead
    """
    def __init__(self, backbone, out_dim=4096):
        super().__init__()
        self.backbone = backbone
        num_ftrs = self.backbone.fc.in_features # 512
        
        # Replace original classifier with identity
        self.backbone.fc = nn.Identity()
        
        # Add DINO-specific head
        self.projection_head = DINOHead(
            in_dim=num_ftrs,
            hidden_dim=2048,
            bottleneck_dim=256,
            out_dim=out_dim
        )

    def forward(self, x):
        features = self.backbone(x)
        projections = self.projection_head(features)
        return projections

# --- Helper functions to build models ---

def get_simclr_model():
    resnet_backbone = models.resnet18(weights=None)
    return SimCLR(backbone=resnet_backbone, proj_dim=128)

def get_byol_model():
    resnet_backbone = models.resnet18(weights=None)
    return BYOL(backbone=resnet_backbone, proj_dim=128)

def get_dino_model():
    resnet_backbone = models.resnet18(weights=None)
    # Output dim (prototypes) is typically large for DINO
    return DINO(backbone=resnet_backbone, out_dim=4096)


# --- Test block ---
if __name__ == '__main__':
    dummy_input = torch.randn(4, 3, 96, 96) # (batch_size, C, H, W)
    
    print("--- Testing Model Definitions ---")
    
    # Test teacher
    teacher_model = get_resnet18_teacher()
    print("\nResNet-18 Teacher built successfully.")
    output = teacher_model(dummy_input)
    print(f"Teacher output shape (classes): {output.shape}") # Should be [4, 10]
    
    # Test SimCLR
    simclr_model = get_simclr_model()
    print("\nSimCLR (ResNet-18 + Proj Head) built successfully.")
    output = simclr_model(dummy_input)
    print(f"SimCLR output shape (projections): {output.shape}") # Should be [4, 128]
    
    # Test BYOL
    byol_model = get_byol_model()
    print("\nBYOL (ResNet-18 + Proj + Pred Head) built successfully.")
    output = byol_model(dummy_input)
    print(f"BYOL output shape (predictions): {output.shape}") # Should be [4, 128]
    
    # Test DINO
    dino_model = get_dino_model()
    print("\nDINO (ResNet-18 + DINO Head) built successfully.")
    output = dino_model(dummy_input)
    print(f"DINO output shape (prototypes): {output.shape}") # Should be [4, 4096]
    
    print("\n--- All models built successfully! ---")