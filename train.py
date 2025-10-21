# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from model import EfficientNetClassifier, ViTClassifier, HybridEfficientTransformer

def get_dataloaders(data_dir, img_size=224, batch_size=16, num_workers=4):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2,0.2,0.2,0.02),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    train_ds = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_tf)
    val_ds = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=val_tf)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_dl, val_dl, train_ds.classes

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for imgs, labels in tqdm(loader, desc="Train"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return running_loss / len(loader.dataset), correct / total

def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_dl, val_dl, classes = get_dataloaders(args.data_dir, img_size=args.img_size, batch_size=args.batch_size, num_workers=args.num_workers)
    num_classes = len(classes)

    # prepare models
    if args.model == 'efficientnet':
        model = EfficientNetClassifier(num_classes, model_name=args.eff_name).to(device)
    elif args.model == 'vit':
        model = ViTClassifier(num_classes, model_name=args.vit_name).to(device)
    elif args.model == 'hybrid':
        model = HybridEfficientTransformer(num_classes, eff_name=args.eff_name, proj_dim=args.proj_dim, nhead=args.nhead, num_layers=args.num_layers).to(device)
    else:
        raise ValueError("model must be one of [efficientnet, vit, hybrid]")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_acc = 0.0
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        train_loss = train_one_epoch(model, train_dl, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_dl, criterion, device)
        print(f"Train Loss: {train_loss:.4f}  |  Val Loss: {val_loss:.4f}  |  Val Acc: {val_acc:.4%}")
        scheduler.step()

        # save best
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model, f'best_{args.model}.pth')
            print("Saved best model.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='dataset', type=str)
    parser.add_argument('--model', default='hybrid', choices=['efficientnet','vit','hybrid'])
    parser.add_argument('--eff_name', default='efficientnet_b0')
    parser.add_argument('--vit_name', default='vit_base_patch16_224')
    parser.add_argument('--img_size', default=224, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--proj_dim', default=256, type=int)
    parser.add_argument('--nhead', default=8, type=int)
    parser.add_argument('--num_layers', default=4, type=int)
    args = parser.parse_args()
    main(args)
