import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from models.daa_mdenet import DAAMDeNet
from dataset import get_dataloaders
from avlo import AVLO

def train_model(data_dir, num_epochs, batch_size, learning_rate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cpu':
        print("WARNING: Training on CPU will be very slow for 125,000 images!")
    
    train_loader, test_loader, classes = get_dataloaders(data_dir, batch_size)
    num_classes = len(classes)
    
    print("Initializing DAA-MDeNet model...")
    model = DAAMDeNet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print("Starting training loop...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Print progress every 50 batches
            if (i + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
            
        train_acc = 100 * correct / total
        
        # Validation
        print(f"Validating Epoch {epoch+1}...")
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_acc = 100 * val_correct / val_total
        print(f"Epoch [{epoch+1}/{num_epochs}] SUMMARY - Train Loss: {running_loss/len(train_loader):.4f} - Train Acc: {train_acc:.2f}% - Val Acc: {val_acc:.2f}%")
        
    return val_acc

def optimize_hyperparameters(data_dir):
    """
    Use AVLO to find the best hyperparameters.
    Bounds: [Epochs (5-20), Batch Size (16-64), LR (0.0001-0.01)]
    """
    bounds = [[5, 20], [16, 64], [0.0001, 0.01]]
    
    def objective_func(params):
        epochs = int(params[0])
        batch_size = int(params[1])
        lr = params[2]
        print(f"Testing params - Epochs: {epochs}, Batch Size: {batch_size}, LR: {lr:.5f}")
        try:
            val_acc = train_model(data_dir, epochs, batch_size, lr)
            return -val_acc # Minimize negative accuracy
        except Exception as e:
            print(f"Error during training: {e}")
            return 0.0
            
    optimizer = AVLO(objective_func, bounds, pop_size=5, max_iter=3)
    best_params, best_fit = optimizer.optimize()
    print(f"Best Parameters: Epochs={int(best_params[0])}, Batch Size={int(best_params[1])}, LR={best_params[2]:.5f}")
    print(f"Best Validation Accuracy: {-best_fit:.2f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plant Disease Detection using DAA-MDeNet")
    parser.add_argument('--data_dir', type=str, default='dataset_flat', help='Path to restructured PlantifyDr dataset')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--optimize', action='store_true', help='Run AVLO hyperparameter optimization')
    
    args = parser.parse_args()
    
    if args.optimize:
        optimize_hyperparameters(args.data_dir)
    else:
        train_model(args.data_dir, args.epochs, args.batch_size, args.lr)
