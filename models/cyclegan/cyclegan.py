import argparse
import os
import numpy as np
import itertools
import time
import sys
from tqdm import tqdm
from PIL import Image

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.utils import save_image

# Local imports
from .model import GeneratorResNet, Discriminator, weights_init_normal
from .datasets import ImageDataset
from .utils import ReplayBuffer, LambdaLR
from core.performanceIterator import PerformanceIterator

def get_transforms(args):
    return [
        transforms.Resize(int(args.img_height * 1.12), Image.BICUBIC),
        transforms.RandomCrop((args.img_height, args.img_width)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

def train(args, device):
    # Create directory for checkpoints
    checkpoint_dir = f"saved_models/{args.dataset_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    input_shape = (args.channels, args.img_height, args.img_width)
    G_AB = GeneratorResNet(input_shape, args.n_residual_blocks).to(device)
    G_BA = GeneratorResNet(input_shape, args.n_residual_blocks).to(device)
    D_A = Discriminator(input_shape).to(device)
    D_B = Discriminator(input_shape).to(device)

    G_AB.apply(weights_init_normal)
    G_BA.apply(weights_init_normal)
    D_A.apply(weights_init_normal)
    D_B.apply(weights_init_normal)

    # Losses & Optimizers
    criterion_GAN = torch.nn.MSELoss().to(device)
    criterion_cycle = torch.nn.L1Loss().to(device)
    criterion_identity = torch.nn.L1Loss().to(device)

    optimizer_G = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=args.alpha, betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=args.alpha, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=args.alpha, betas=(0.5, 0.999))

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(args.n_epochs, 0, args.decay_epoch).step)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(args.n_epochs, 0, args.decay_epoch).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(args.n_epochs, 0, args.decay_epoch).step)

    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    dataset_path = f"./data/{args.dataset_name}"
    dataloader = DataLoader(
        ImageDataset(dataset_path, transforms_=get_transforms(args), unaligned=True),
        batch_size=args.batch_size, shuffle=True, num_workers=4
    )

    if args.enable_perf_log:
        dataloader = PerformanceIterator(dataloader, None, None, None, args.log_file)

    total_steps = 0
    Tensor = torch.cuda.FloatTensor if device.type == 'cuda' else torch.Tensor

    print(f"Starting CycleGAN Training on {device}...")

    while total_steps < args.num_steps:
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="CycleGAN Training"):
            if total_steps >= args.num_steps: break

            real_A = Variable(batch["A"].type(Tensor).to(device))
            real_B = Variable(batch["B"].type(Tensor).to(device))
            valid = Variable(Tensor(np.ones((real_A.size(0), *D_A.output_shape))).to(device), requires_grad=False)
            fake = Variable(Tensor(np.zeros((real_A.size(0), *D_A.output_shape))).to(device), requires_grad=False)

            # Generator Step
            # Identity loss
            optimizer_G.zero_grad()
            loss_id_A = criterion_identity(G_BA(real_A), real_A)
            loss_id_B = criterion_identity(G_AB(real_B), real_B)
            loss_identity = (loss_id_A + loss_id_B) / 2
            # GAN Loss
            fake_B = G_AB(real_A); loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
            fake_A = G_BA(real_B); loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)
            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2
            # Cycle loss
            recov_A = G_BA(fake_B); loss_cycle_A = criterion_cycle(recov_A, real_A)
            recov_B = G_AB(fake_A); loss_cycle_B = criterion_cycle(recov_B, real_B)
            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2
            # Total loss
            loss_G = loss_GAN + 10.0 * loss_cycle + 5.0 * loss_identity
            loss_G.backward(); optimizer_G.step()

            # Discriminator Steps
            optimizer_D_A.zero_grad()
            loss_D_A = (criterion_GAN(D_A(real_A), valid) + criterion_GAN(D_A(fake_A_buffer.push_and_pop(fake_A).detach()), fake)) / 2
            loss_D_A.backward(); optimizer_D_A.step()

            optimizer_D_B.zero_grad()
            loss_D_B = (criterion_GAN(D_B(real_B), valid) + criterion_GAN(D_B(fake_B_buffer.push_and_pop(fake_B).detach()), fake)) / 2
            loss_D_B.backward(); optimizer_D_B.step()

            total_steps += 1
        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()
    
    # Save model checkpoints
    print(f"Saving models to {checkpoint_dir}...")
    torch.save(G_AB.state_dict(), f"{checkpoint_dir}/G_AB.pth")
    torch.save(G_BA.state_dict(), f"{checkpoint_dir}/G_BA.pth")
    torch.save(D_A.state_dict(), f"{checkpoint_dir}/D_A.pth")
    torch.save(D_B.state_dict(), f"{checkpoint_dir}/D_B.pth")

def test(args, device):
    print(f"Performing CycleGAN Inference on {device}...")
    input_shape = (args.channels, args.img_height, args.img_width)
    G_AB = GeneratorResNet(input_shape, args.n_residual_blocks).to(device)
    G_BA = GeneratorResNet(input_shape, args.n_residual_blocks).to(device)

    # Standardized checkpoint path
    G_AB.load_state_dict(torch.load(f"saved_models/{args.dataset_name}/G_AB.pth", map_location=device))
    G_BA.load_state_dict(torch.load(f"saved_models/{args.dataset_name}/G_BA.pth", map_location=device))
    G_AB.eval(); G_BA.eval()

    test_dataloader = DataLoader(
        ImageDataset(f"./data/{args.dataset_name}", transforms_=get_transforms(args), unaligned=True, mode="test"),
        batch_size=args.batch_size, shuffle=False, num_workers=1
    )

    if args.enable_perf_log:
        test_dataloader = PerformanceIterator(test_dataloader, None, None, None, args.log_file)

    os.makedirs(f"output/{args.dataset_name}", exist_ok=True)
    total_steps = 0
    Tensor = torch.cuda.FloatTensor if device.type == 'cuda' else torch.Tensor

    with torch.no_grad():
        while total_steps < args.num_steps:
            for i, batch in enumerate(test_dataloader):
                if total_steps >= args.num_steps: break
                real_A = Variable(batch["A"].type(Tensor).to(device))
                fake_B = G_AB(real_A)
                save_image(fake_B, f"output/{args.dataset_name}/{total_steps}.png", normalize=True)
                total_steps += 1    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standardized CycleGAN Benchmark")
    parser.add_argument("--gpuIdx", type=int, default=0)
    parser.add_argument("--job_type", type=str, choices=['training', 'inference'], default='training')
    parser.add_argument("--dataset_name", type=str, default="monet2photo")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.0002)
    parser.add_argument("--img_height", type=int, default=256)
    parser.add_argument("--img_width", type=int, default=256)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--n_residual_blocks", type=int, default=9)
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--enable_perf_log", action='store_true')
    parser.add_argument("--log_file", type=str, default="cyclegan.log")
    args = parser.parse_args()

    DEVICE = torch.device(f"cuda:{args.gpuIdx}" if torch.cuda.is_available() else "cpu")
    if args.job_type == "training": train(args, DEVICE)
    else: test(args, DEVICE)