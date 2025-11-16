import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Image size as specified in the proposal (96x96)
IMG_SIZE = 96

# Normalization values for STL-10
# These are standard for models pretrained on ImageNet, and are a good default
NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

class SSLTransform:
    """
    Applies strong data augmentation twice to create two "views" for 
    self-supervised learning methods like SimCLR and BYOL.
    """
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))
            ], p=0.5),
            NORMALIZE
        ])

    def __call__(self, x):
        # Apply the transform twice to get two augmented views
        view1 = self.transform(x)
        view2 = self.transform(x)
        return view1, view2

def get_supervised_transform():
    """
    Standard transformations for supervised training and evaluation.
    - Resize to 96x96
    - Convert to Tensor
    - Normalize
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        NORMALIZE
    ])

def get_stl10_dataloaders(batch_size, ssl_transform=False):
    """
    Main function to get the STL-10 DataLoaders.
    
    Args:
        batch_size (int): The batch size for the data loaders.
        ssl_transform (bool): If True, returns the unlabeled loader with
                              SSL-specific transforms. Otherwise, uses
                              standard supervised transforms.
    
    Returns:
        tuple: (unlabeled_loader, train_loader, test_loader)
               unlabeled_loader will be None if ssl_transform=False.
    """
    
    # 1. Labeled Training Set (for fine-tuning and KD)
    train_dataset = datasets.STL10(
        root='./data',
        split='train',
        download=True,
        transform=get_supervised_transform()
    )
    
    # 2. Test Set (for final evaluation)
    test_dataset = datasets.STL10(
        root='./data',
        split='test',
        download=True,
        transform=get_supervised_transform()
    )
    
    unlabeled_loader = None
    if ssl_transform:
        # 3. Unlabeled Set (for SSL pretraining)
        unlabeled_dataset = datasets.STL10(
            root='./data',
            split='unlabeled',
            download=True,
            transform=SSLTransform()
        )
        unlabeled_loader = DataLoader(
            dataset=unlabeled_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=True # Important for SSL
        )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False, # No need to shuffle test data
        num_workers=0,
        pin_memory=True
    )
    
    print("--- DataLoaders Created ---")
    if unlabeled_loader:
        print(f"Unlabeled (SSL) batches: {len(unlabeled_loader)}")
    print(f"Labeled Train batches: {len(train_loader)}")
    print(f"Labeled Test batches:  {len(test_loader)}")
    print("---------------------------")
    
    return unlabeled_loader, train_loader, test_loader

if __name__ == '__main__':
    # You can run this file directly to test it:
    # python -m utils.dataset
    
    print("Testing Supervised DataLoaders...")
    _, train_loader, test_loader = get_stl10_dataloaders(batch_size=64, ssl_transform=False)
    
    # Check one batch
    imgs, labels = next(iter(train_loader))
    print(f"Supervised batch shape: {imgs.shape}, Labels shape: {labels.shape}")

    print("\nTesting SSL DataLoaders...")
    unlabeled_loader, _, _ = get_stl10_dataloaders(batch_size=64, ssl_transform=True)
    
    # Check one batch
    (view1, view2), _ = next(iter(unlabeled_loader)) # Labels are returned but are -1 (unused)
    print(f"SSL View 1 shape: {view1.shape}, View 2 shape: {view2.shape}")