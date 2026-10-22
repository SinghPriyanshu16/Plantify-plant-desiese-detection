import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(data_dir, batch_size=32):
    """
    Loads the PlantifyDr dataset from the flattened directory structure.
    Expected structure:
    dataset_flat/
        train/
            class1/
            class2/
        valid/
            class1/
            class2/
    """
    print("Preparing data transformations...")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # Standard normalization for ImageNet-like data
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory '{train_dir}' not found.")

    print(f"Loading training dataset from {train_dir}...")
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    
    print(f"Loading validation dataset from {valid_dir}...")
    test_dataset = datasets.ImageFolder(root=valid_dir, transform=transform)
    
    # num_workers=0 is safer on Windows and prevents hanging
    print("Creating DataLoaders...")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Found {len(train_dataset)} training images and {len(test_dataset)} validation images across {len(train_dataset.classes)} classes.")
    return train_loader, test_loader, train_dataset.classes
