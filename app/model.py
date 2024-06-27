import torch
import torch.nn as nn
import torchvision.models as models

class MultiTaskModel(nn.Module):
    def __init__(self, num_nationalities, num_emotions, num_dress_colors):
        super(MultiTaskModel, self).__init__()
        self.base_model = models.resnet50(pretrained=True)
        num_ftrs = self.base_model.fc.in_features
        
        self.base_model.fc = nn.Identity()
        
        self.nationality_fc = nn.Linear(num_ftrs, num_nationalities)
        self.emotion_fc = nn.Linear(num_ftrs, num_emotions)
        self.age_fc = nn.Linear(num_ftrs, 1)
        self.dress_color_fc = nn.Linear(num_ftrs, num_dress_colors)

    def forward(self, x):
        features = self.base_model(x)
        
        nationality = self.nationality_fc(features)
        emotion = self.emotion_fc(features)
        age = self.age_fc(features)
        dress_color = self.dress_color_fc(features)
        
        return nationality, emotion, age, dress_color
