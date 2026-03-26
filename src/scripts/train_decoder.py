"""
Train LeWM-VC Encoder-Decoder

Trains the encoder-decoder pair on PEViD-HD video data.
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class PEViDDataset(Dataset):
    """PEViD-HD video dataset for training."""

    def __init__(self, video_paths, frame_size=(256, 256), frames_per_clip=16):
        self.videos = []
        self.frame_size = frame_size
        self.frames_per_clip = frames_per_clip

        for path in video_paths:
            if Path(path).exists():
                self.videos.append(path)

        print(f"Loaded {len(self.videos)} videos")

    def __len__(self):
        return len(self.videos) * 100

    def __getitem__(self, idx):
        video_idx = idx % len(self.videos)
        cap = cv2.VideoCapture(self.videos[video_idx])

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start_frame = np.random.randint(0, max(1, total_frames - self.frames_per_clip))

        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for _ in range(self.frames_per_clip):
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()

            frame = cv2.resize(frame, self.frame_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0
            frame = np.transpose(frame, (2, 0, 1))
            frames.append(frame)

        cap.release()

        frames = np.stack(frames, axis=0)
        return torch.from_numpy(frames).float()


class VideoAutoencoder(nn.Module):
    """Full encoder-decoder for video."""

    def __init__(self, latent_dim=192):
        super().__init__()

        from lewm_vc.encoder import LeWMEncoder
        from lewm_vc.working_decoder import LeWMDecoder

        self.encoder = LeWMEncoder(latent_dim=latent_dim, semantic_surprise=True)
        self.decoder = LeWMDecoder(latent_dim=latent_dim)

    def forward(self, x):
        batch_size, num_frames, c, h, w = x.shape

        x = x.view(batch_size * num_frames, c, h, w)

        latent = self.encoder(x, return_surprise=False)
        reconstructed = self.decoder(latent)

        reconstructed = reconstructed.view(batch_size, num_frames, c, h, w)

        return reconstructed, latent


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    video_paths = list(Path("datasets/pevid-hd").glob("*.mpg"))
    if not video_paths:
        video_paths = list(Path("datasets/pevid-hd").glob("*.mp4"))

    if not video_paths:
        print("No videos found in datasets/pevid-hd/")
        return

    dataset = PEViDDataset(
        [str(p) for p in video_paths],
        frame_size=(256, 256),
        frames_per_clip=args.frames_per_clip
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    model = VideoAutoencoder(latent_dim=args.latent_dim).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    criterion = nn.MSELoss()

    os.makedirs("checkpoints", exist_ok=True)

    print(f"\nTraining for {args.epochs} epochs...")
    print(f"Frames per clip: {args.frames_per_clip}")
    print(f"Batch size: {args.batch_size}")
    print(f"Latent dim: {args.latent_dim}")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            batch = batch.to(device)

            optimizer.zero_grad()

            reconstructed, latent = model(batch)

            loss = criterion(reconstructed, batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, LR = {scheduler.get_last_lr()[0]:.6f}")

        if (epoch + 1) % 5 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
            }, f"checkpoints/autoencoder_epoch_{epoch+1}.pt")
            print(f"Saved checkpoint at epoch {epoch+1}")

    torch.save(model.state_dict(), "checkpoints/autoencoder_final.pt")
    print("Training complete!")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--latent_dim", type=int, default=192)
    parser.add_argument("--frames_per_clip", type=int, default=16)

    args = parser.parse_args()
    train(args)
