import torch
from torchvision import transforms
from PIL import Image

def load_model(model_path, num_nationalities, num_emotions, num_dress_colors):
    from app.model import MultiTaskModel
    model = MultiTaskModel(num_nationalities, num_emotions, num_dress_colors)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def preprocess_image(image_path, img_size=224):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def predict(model, image_tensor):
    with torch.no_grad():
        nationality, emotion, age, dress_color = model(image_tensor)
        
        nationality = nationality.argmax(1).item()
        emotion = emotion.argmax(1).item()
        age = age.item()
        dress_color = dress_color.argmax(1).item()
        
    return nationality, emotion, age, dress_color

def save_model(model, path):
    torch.save(model.state_dict(), path)