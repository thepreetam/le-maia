# LeWM-VC GPU Training on Google Colab
# Runtime: Runtime -> Change runtime type -> GPU

# Install dependencies
!pip install torch torchvision numpy opencv-python tqdm

# Clone the repo
!git clone https://github.com/thepreetam/le-maia.git
%cd le-maia

import sys
sys.path.insert(0, 'src')

import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Download sample videos for training
print("Downloading sample videos...")
!mkdir -p datasets/pevid-hd

# Download Big Buck Bunny (Blender Foundation, free to use)
print("Downloading Big Buck Bunny...")
!cd datasets/pevid-hd && wget -q "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4" -O bbb.mp4

# Download Tears of Steel
print("Downloading Tears of Steel...")
!cd datasets/pevid-hd && wget -q "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/TearsOfSteel.mp4" -O tos.mp4

# List what we got
!echo "=== Videos downloaded ==="
!ls -lh datasets/pevid-hd/

print(f"\nCUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# Import LeWM modules
from lewm_vc.encoder import LeWMEncoder
from lewm_vc.working_decoder import LeWMDecoder


class VideoDataset(Dataset):
    def __init__(self, video_paths, frame_size=(256, 256), frames_per_clip=8):
        self.videos = video_paths
        self.frame_size = frame_size
        self.frames_per_clip = frames_per_clip
    
    def __len__(self):
        return len(self.videos) * 100
    
    def __getitem__(self, idx):
        video_idx = idx % len(self.videos)
        cap = cv2.VideoCapture(self.videos[video_idx])
        
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start = np.random.randint(0, max(1, total - self.frames_per_clip))
        
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        
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
        return torch.from_numpy(np.stack(frames)).float()


class VideoAutoencoder(nn.Module):
    def __init__(self, latent_dim=192):
        super().__init__()
        self.encoder = LeWMEncoder(latent_dim=latent_dim, semantic_surprise=True)
        self.decoder = LeWMDecoder(latent_dim=latent_dim)
    
    def forward(self, x):
        batch_size, num_frames, c, h, w = x.shape
        x_flat = x.view(batch_size * num_frames, c, h, w)
        latent = self.encoder(x_flat, return_surprise=False)
        reconstructed = self.decoder(latent, target_size=(h, w))
        reconstructed = reconstructed.view(batch_size, num_frames, c, h, w)
        return reconstructed, latent


def train(epochs=100, batch_size=8, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    
    import glob
    video_paths = glob.glob('datasets/pevid-hd/*.mp4') + glob.glob('datasets/pevid-hd/*.avi') + glob.glob('datasets/pevid-hd/*.mkv')
    
    if not video_paths:
        print("No videos found!")
        return
    
    print(f"Found {len(video_paths)} videos: {video_paths}")
    
    dataset = VideoDataset(video_paths, frame_size=(256, 256), frames_per_clip=4)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    model = VideoAutoencoder(latent_dim=192).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.L1Loss()
    
    import os
    os.makedirs('checkpoints', exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
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
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        scheduler.step()
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")
        
        if (epoch + 1) % 10 == 0:
            torch.save({
                'encoder': model.encoder.state_dict(),
                'decoder': model.decoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'loss': avg_loss,
            }, f'checkpoints/autoencoder_e{epoch+1}.pt')
    
    torch.save({
        'encoder': model.encoder.state_dict(),
        'decoder': model.decoder.state_dict(),
    }, 'checkpoints/autoencoder_final.pt')
    
    print("Training complete!")
    from google.colab import files
    for f in sorted(glob.glob('checkpoints/*.pt')):
        files.download(f)


if __name__ == "__main__":
    train(epochs=100, batch_size=8, lr=1e-4)
