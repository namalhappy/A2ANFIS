import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.spectral_norm as spectral_norm
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import itertools
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tqdm import tqdm
import time

# ---------------- MODULES ---------------- #

class ChannelAttention(nn.Module):
    """Channel-wise attention mechanism for feature weighting."""
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class AnfisLayer(nn.Module):
    """Adaptive Neuro-Fuzzy Inference System Layer."""
    def __init__(self, num_inputs, num_membership_functions):
        super(AnfisLayer, self).__init__()
        self.num_inputs = num_inputs
        self.n_mf = num_membership_functions
        self.num_rules = num_membership_functions ** num_inputs
        self.mu = nn.Parameter(torch.randn(num_inputs, num_membership_functions))
        self.sigma = nn.Parameter(torch.ones(num_inputs, num_membership_functions))
        self.consequent_weights = nn.Parameter(torch.randn(self.num_rules, num_inputs))
        self.consequent_bias = nn.Parameter(torch.randn(self.num_rules))

    def forward(self, x):
        batch_size = x.size(0)
        x_expand = x.unsqueeze(2)
        # Gaussian Membership Function
        membership = torch.exp(-torch.pow(x_expand - self.mu, 2) / (2 * torch.pow(torch.abs(self.sigma) + 1e-5, 2)))
        mf_list = [membership[:, i, :] for i in range(self.num_inputs)]
        indices = list(itertools.product(range(self.n_mf), repeat=self.num_inputs))
        
        rule_strengths = []
        for idx_tuple in indices:
            rule_val = torch.ones(batch_size, device=x.device)
            for input_idx, mf_idx in enumerate(idx_tuple):
                rule_val = rule_val * mf_list[input_idx][:, mf_idx]
            rule_strengths.append(rule_val)
            
        w = torch.stack(rule_strengths, dim=1) 
        w_sum = torch.sum(w, dim=1, keepdim=True)
        w_norm = w / (w_sum + 1e-6) 
        
        x_in = x.unsqueeze(1) 
        coeffs = self.consequent_weights.unsqueeze(0) 
        rule_outputs = torch.sum(x_in * coeffs, dim=2) + self.consequent_bias 
        return torch.sum(w_norm * rule_outputs, dim=1, keepdim=True)

class HybridGenerator(nn.Module):
    def __init__(self, in_channels):
        super(HybridGenerator, self).__init__()
        self.encoder_block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1), 
            nn.BatchNorm2d(64), 
            nn.LeakyReLU(0.2),
            ChannelAttention(64), 
            nn.Conv2d(64, 128, 3, padding=1), 
            nn.BatchNorm2d(128), 
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2)
        )
        self.encoder_block2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), 
            nn.BatchNorm2d(256), 
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 3, padding=1), 
            nn.BatchNorm2d(512), 
            nn.LeakyReLU(0.2),
            ChannelAttention(512), 
            nn.MaxPool2d(2)
        )
        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 3 * 3, 256), 
            nn.LeakyReLU(0.2),
            nn.Linear(256, 6), 
            nn.Tanh() 
        )
        self.anfis = AnfisLayer(num_inputs=6, num_membership_functions=2)

    def forward(self, x):
        x = self.encoder_block1(x)
        x = self.encoder_block2(x)
        features = self.bottleneck(x)
        return self.anfis(features)

class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()
        def sn_conv(in_c, out_c, k, s, p):
            return spectral_norm(nn.Conv2d(in_c, out_c, k, stride=s, padding=p))
        
        self.feat_extract = nn.Sequential(
            sn_conv(in_channels, 64, 3, 2, 1), 
            nn.LeakyReLU(0.2),
            sn_conv(64, 128, 3, 2, 1), 
            nn.BatchNorm2d(128), 
            nn.LeakyReLU(0.2),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            spectral_norm(nn.Linear(2048 + 1, 512)), 
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(512, 1)), 
            nn.Sigmoid()
        )
    def forward(self, x, y):
        feats = self.feat_extract(x)
        combined = torch.cat([feats, y], dim=1)
        return self.classifier(combined)

# ---------------- UTILS ---------------- #

class CycloneDataset(Dataset):
    def __init__(self, h5_path, stats=None):
        try:
            with h5py.File(h5_path, "r") as f:
                self.x_data = f["qvapor_cube"][:]
                raw_y = f["conv_strength"][:]
                self.y_data = np.log1p(raw_y)
                
                if stats is None:
                    self.mean = np.mean(self.x_data)
                    self.std = np.std(self.x_data)
                else:
                    self.mean, self.std = stats
                
                self.x_data = (self.x_data - self.mean) / (self.std + 1e-6)
                self.x_data = torch.from_numpy(self.x_data).float()
                self.y_data = torch.from_numpy(self.y_data).float()
                self.length = len(self.y_data)
        except Exception as e:
            print(f"Error loading {h5_path}: {e}")
            raise

    def __len__(self): return self.length
    def get_stats(self): return self.mean, self.std
    def __getitem__(self, idx): return self.x_data[idx], self.y_data[idx]

def weighted_l1_loss(pred, target, weight_val=5.0):
    l1 = torch.abs(pred - target)
    weight_mask = torch.ones_like(target)
    weight_mask[target > 2.0] = weight_val 
    return torch.mean(l1 * weight_mask)

# ---------------- MAIN ---------------- #

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    # Data
    train_dataset = CycloneDataset(args.train_file, stats=None)
    train_stats = train_dataset.get_stats()
    test_dataset = CycloneDataset(args.test_file, stats=train_stats)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Models
    netG = HybridGenerator(args.channels).to(device)
    netD = Discriminator(args.channels).to(device)
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr)
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr)
    criterionGAN = nn.BCELoss()

    # Simple Training Loop (Condensed for brevity)
    for epoch in range(args.epochs):
        netG.train(); netD.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for x, y in loop:
            x, y = x.to(device), y.to(device)
            
            # Update D
            optimizerD.zero_grad()
            real_labels = torch.ones(x.size(0), 1, device=device)
            fake_labels = torch.zeros(x.size(0), 1, device=device)
            
            loss_d_real = criterionGAN(netD(x, y), real_labels)
            loss_d_fake = criterionGAN(netD(x, netG(x).detach()), fake_labels)
            (loss_d_real + loss_d_fake).backward()
            optimizerD.step()
            
            # Update G
            optimizerG.zero_grad()
            fake_y = netG(x)
            loss_g_gan = criterionGAN(netD(x, fake_y), real_labels)
            loss_g_l1 = weighted_l1_loss(fake_y, y) * 100.0
            (loss_g_gan + loss_g_l1).backward()
            optimizerG.step()

    # Save and Evaluate
    torch.save(netG.state_dict(), os.path.join(args.model_dir, "A2ANFIS_model.pth"))
    
    # Inference Timing
    netG.eval()
    y_true, y_pred = [], []
    start_time = time.time()
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            p = netG(x).cpu().numpy()
            y_true.extend(np.expm1(y.numpy().flatten()))
            y_pred.extend(np.maximum(np.expm1(p.flatten()), 0))
    
    inf_time = ((time.time() - start_time) / len(test_dataset)) * 1000
    print(f"Inference complete. Avg time: {inf_time:.4f}ms/sample")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A2ANFIS: Attention-based ANFIS for Cyclone Strength Prediction")
    parser.add_argument("--train_file", type=str, required=True, help="Path to train.h5")
    parser.add_argument("--test_file", type=str, required=True, help="Path to test.h5")
    parser.add_argument("--model_dir", type=str, default="./models", help="Directory to save models")
    parser.add_argument("--results_dir", type=str, default="./results", help="Directory to save results")
    parser.add_argument("--channels", type=int, default=42, help="Number of input channels")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0002)
    
    args = parser.parse_args()
    main(args)
