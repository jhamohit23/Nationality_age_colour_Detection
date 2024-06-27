import time
import logging
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import sys
import os
import numpy as np

# Set the project root to D drive
PROJECT_ROOT = 'D:/multitask_prediction'
sys.path.insert(0, PROJECT_ROOT)

from app.model import MultiTaskModel
from app.dataset import CustomImageDataset

def train_model():
    # Constants
    IMG_SIZE = 224
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001

    # Set up logging
    log_dir = os.path.join(PROJECT_ROOT, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(log_dir, 'training.log'), level=logging.INFO)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data loading and preprocessing
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = CustomImageDataset(
        csv_file=os.path.join(PROJECT_ROOT, 'data', 'custom_dataset_annotations.csv'),
        img_dir=os.path.join(PROJECT_ROOT, 'data', 'custom_dataset'),
        transform=transform
    )
    train_dataset, val_dataset = random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize the model
    num_nationalities = len(dataset.nationality_encoder.classes_)
    num_emotions = len(dataset.emotion_encoder.classes_)
    num_dress_colors = len(dataset.dress_color_encoder.classes_)

    model = MultiTaskModel(num_nationalities, num_emotions, num_dress_colors)
    model = model.to(device)

    # Define loss functions and optimizer
    nationality_criterion = nn.CrossEntropyLoss()
    emotion_criterion = nn.CrossEntropyLoss()
    age_criterion = nn.MSELoss()
    dress_color_criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        
        model.train()
        train_loss = 0.0
        
        for images, nationalities, emotions, ages, dress_colors in train_loader:
            images, nationalities, emotions, ages, dress_colors = (
                images.to(device),
                nationalities.to(device).long(),
                emotions.to(device).long(),
                ages.to(device).float(),
                dress_colors.to(device).long()
            )
            
            optimizer.zero_grad()
            
            nat_pred, emo_pred, age_pred, dress_pred = model(images)
            
            nat_loss = nationality_criterion(nat_pred, nationalities)
            emo_loss = emotion_criterion(emo_pred, emotions)
            age_loss = age_criterion(age_pred.squeeze(), ages)
            dress_loss = dress_color_criterion(dress_pred, dress_colors)
            
            loss = nat_loss + emo_loss + age_loss + dress_loss
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, nationalities, emotions, ages, dress_colors in val_loader:
                images, nationalities, emotions, ages, dress_colors = (
                    images.to(device),
                    nationalities.to(device).long(),
                    emotions.to(device).long(),
                    ages.to(device).float(),
                    dress_colors.to(device).long()
                )
                
                nat_pred, emo_pred, age_pred, dress_pred = model(images)
                
                nat_loss = nationality_criterion(nat_pred, nationalities)
                emo_loss = emotion_criterion(emo_pred, emotions)
                age_loss = age_criterion(age_pred.squeeze(), ages)
                dress_loss = dress_color_criterion(dress_pred, dress_colors)
                
                loss = nat_loss + emo_loss + age_loss + dress_loss
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        end_time = time.time()
        epoch_duration = end_time - start_time
        
        logging.info(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Time: {epoch_duration:.2f}s")

    # Save the model
    model_dir = os.path.join(PROJECT_ROOT, 'models')
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, 'multitask_model.pth'))

if __name__ == "__main__":
    train_model()