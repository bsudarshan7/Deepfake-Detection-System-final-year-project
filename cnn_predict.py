import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.mobilenet_v2()
model.classifier[1] = nn.Linear(model.last_channel, 2)

model.load_state_dict(torch.load("cnn_model.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

def predict_cnn(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)

    probs = torch.softmax(output[0], dim=0)

    confidence = torch.max(probs).item() * 100
    prediction = torch.argmax(probs).item()

    # 0 = real, 1 = fake (depends on folder order)
    return ("FAKE", confidence) if prediction == 1 else ("REAL", confidence)