
import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import medmnist
from medmnist import PathMNIST, INFO
DATASET   = "pathmnist"
DATA_FLAG = "pathmnist"
BATCH     = 64
EPOCHS    = 20
LR        = 1e-3
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR  = "saved_models"
NUM_CLASSES = 9         
IMG_SIZE  = 28    
print(f"Using device: {DEVICE}")
def get_transforms(pretrained=False):

    if pretrained:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
def get_loaders(pretrained=False):
   
    tf = get_transforms(pretrained)
    train_ds = PathMNIST(split="train", transform=tf, download=True)
    val_ds   = PathMNIST(split="val",   transform=tf, download=True)
    test_ds  = PathMNIST(split="test",  transform=tf, download=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False, num_workers=2)
    return train_loader, val_loader, test_loader
clss SimpleCNN(nn.Module):
   
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            nn.Dropout2d(0.25),

           
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),  
            nn.Dropout2d(0.25),

       
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2),  
            nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class DeepCNN(nn.Module):

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        )

        self.res1 = self._res_block(64, 128, downsample=True)   # → 14×14
        self.res2 = self._res_block(128, 256, downsample=True)  # → 7×7
        self.res3 = self._res_block(256, 256, downsample=False) # → 7×7

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    @staticmethod
    def _res_block(in_ch, out_ch, downsample):
        stride = 2 if downsample else 1
        main = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
        )
        shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride),
            nn.BatchNorm2d(out_ch),
        ) if downsample or in_ch != out_ch else nn.Identity()

        class ResBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.main = main
                self.shortcut = shortcut
                self.relu = nn.ReLU()
            def forward(self, x):
                return self.relu(self.main(x) + self.shortcut(x))

        return ResBlock()

    def forward(self, x):
        x = self.stem(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.pool(x)
        return self.classifier(x)


def get_resnet18(num_classes=NUM_CLASSES, freeze_backbone=True):
    
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_efficientnet_b0(num_classes=NUM_CLASSES, freeze_backbone=True):
    
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def unfreeze_resnet_layers(model, num_blocks=2):
 
    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False
    # Then unfreeze layer4, layer3, fc (depending on num_blocks)
    layers_to_unfreeze = [model.fc]
    if num_blocks >= 1:
        layers_to_unfreeze.append(model.layer4)
    if num_blocks >= 2:
        layers_to_unfreeze.append(model.layer3)
    for layer in layers_to_unfreeze:
        for param in layer.parameters():
            param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    return model




def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs   = imgs.to(DEVICE)
        labels = labels.squeeze(1).long().to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += imgs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs   = imgs.to(DEVICE)
        labels = labels.squeeze(1).long().to(DEVICE)

        outputs = model(imgs)
        loss    = criterion(outputs, labels)

        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


def train_model(model, train_loader, val_loader, name, epochs=EPOCHS, lr=LR):
  
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc, best_state = 0.0, None
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer)
        va_loss, va_acc, _, _ = eval_epoch(model, val_loader, criterion)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"[{name}] Epoch {epoch:02d}/{epochs} | "
              f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | "
              f"Val Loss: {va_loss:.4f} Acc: {va_acc:.4f}")

    elapsed = time.time() - start_time
    history["train_time_sec"] = elapsed
    history["best_val_acc"]   = best_val_acc

    # Save best model
    model.load_state_dict(best_state)
    os.makedirs(SAVE_DIR, exist_ok=True)
    torch.save(model.state_dict(), f"{SAVE_DIR}/{name}.pth")
    print(f"  ✓ Saved {name}.pth  |  Best Val Acc: {best_val_acc:.4f}  |  Time: {elapsed:.0f}s\n")
    return model, history



def plot_history(histories, save_path="results/training_curves.png"):
 
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    n = len(histories)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 8))
    if n == 1:
        axes = axes.reshape(2, 1)

    for i, (name, hist) in enumerate(histories.items()):
        ep = range(1, len(hist["train_loss"]) + 1)
        # Loss
        axes[0, i].plot(ep, hist["train_loss"], label="Train")
        axes[0, i].plot(ep, hist["val_loss"],   label="Val")
        axes[0, i].set_title(f"{name}\nLoss")
        axes[0, i].legend(); axes[0, i].set_xlabel("Epoch")
        # Accuracy
        axes[1, i].plot(ep, hist["train_acc"], label="Train")
        axes[1, i].plot(ep, hist["val_acc"],   label="Val")
        axes[1, i].set_title(f"{name}\nAccuracy")
        axes[1, i].legend(); axes[1, i].set_xlabel("Epoch")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"  ✓ Curves saved → {save_path}")


def evaluate_and_report(model, test_loader, name, class_names):
   
    criterion = nn.CrossEntropyLoss()
    _, acc, preds, labels = eval_epoch(model, test_loader, criterion)
    report = classification_report(labels, preds,
                                   target_names=class_names, output_dict=True)
    print(f"\n── Test Report: {name} ──")
    print(classification_report(labels, preds, target_names=class_names))
    return {"accuracy": acc, "report": report}


if __name__ == "__main__":
    CLASS_NAMES = [
        "Adipose", "Background", "Debris",
        "Lymphocytes", "Mucus", "Smooth Muscle",
        "Normal Colon Mucosa", "Cancer-Assoc. Stroma", "Colorectal Adenocarcinoma"
    ]

    all_histories = {}
    all_results   = {}

    
    print("\n" + "="*60)
    print(" 1a. SimpleCNN (from scratch)")
    print("="*60)
    train_l, val_l, test_l = get_loaders(pretrained=False)
    model_a, hist_a = train_model(SimpleCNN(), train_l, val_l, "SimpleCNN")
    all_histories["SimpleCNN"] = hist_a
    all_results["SimpleCNN"]   = evaluate_and_report(model_a, test_l, "SimpleCNN", CLASS_NAMES)

    
    print("\n" + "="*60)
    print(" 1b. DeepCNN / ResidualCNN (from scratch)")
    print("="*60)
    model_b, hist_b = train_model(DeepCNN(), train_l, val_l, "DeepCNN")
    all_histories["DeepCNN"] = hist_b
    all_results["DeepCNN"]   = evaluate_and_report(model_b, test_l, "DeepCNN", CLASS_NAMES)

    print("\n" + "="*60)
    print(" 2a. Transfer Learning — ResNet-18 (frozen backbone)")
    print("="*60)
    train_l_pt, val_l_pt, test_l_pt = get_loaders(pretrained=True)
    model_c, hist_c = train_model(get_resnet18(freeze_backbone=True),
                                  train_l_pt, val_l_pt, "ResNet18_TL")
    all_histories["ResNet18_TL"] = hist_c
    all_results["ResNet18_TL"]   = evaluate_and_report(model_c, test_l_pt, "ResNet18_TL", CLASS_NAMES)

  
    print("\n" + "="*60)
    print(" 2b. Transfer Learning — EfficientNet-B0 (frozen backbone)")
    print("="*60)
    model_d, hist_d = train_model(get_efficientnet_b0(freeze_backbone=True),
                                  train_l_pt, val_l_pt, "EfficientNetB0_TL")
    all_histories["EfficientNetB0_TL"] = hist_d
    all_results["EfficientNetB0_TL"]   = evaluate_and_report(model_d, test_l_pt,
                                                              "EfficientNetB0_TL", CLASS_NAMES)


    print("\n" + "="*60)
    print(" 3. Fine-Tuning — ResNet-18 (layers 3+4 unfrozen)")
    print("="*60)
    ft_model = get_resnet18(freeze_backbone=True)
    ft_model  = unfreeze_resnet_layers(ft_model, num_blocks=2)
    model_e, hist_e = train_model(ft_model, train_l_pt, val_l_pt,
                                  "ResNet18_FT", lr=5e-4)
    all_histories["ResNet18_FT"] = hist_e
    all_results["ResNet18_FT"]   = evaluate_and_report(model_e, test_l_pt,
                                                        "ResNet18_FT", CLASS_NAMES)

    
    plot_history(all_histories, "results/training_curves.png")

    print("\n" + "="*60)
    print(" SUMMARY TABLE")
    print("="*60)
    print(f"{'Model':<25} {'Test Acc':>10} {'Time (s)':>10} {'Best Val Acc':>14}")
    print("-"*62)
    for name, res in all_results.items():
        t = all_histories[name]["train_time_sec"]
        v = all_histories[name]["best_val_acc"]
        print(f"{name:<25} {res['accuracy']:>10.4f} {t:>10.0f} {v:>14.4f}")


    with open("results/results_summary.json", "w") as f:
        summary = {k: {"accuracy": v["accuracy"],
                       "best_val_acc": all_histories[k]["best_val_acc"],
                       "train_time_sec": all_histories[k]["train_time_sec"]}
                   for k, v in all_results.items()}
        json.dump(summary, f, indent=2)
    print("\n✓ All done! Results saved to results/results_summary.json")
