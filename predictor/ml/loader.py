import torch
import json
from torchvision import transforms
from PIL import Image
from torchvision.models import efficientnet_b3

MODEL_PATH = "predictor/ml/best_rice_model.pth"
CLASS_NAMES_PATH = "predictor/ml/class_names.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load class names from JSON
with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

num_classes = len(class_names)

# ---- RECREATE EfficientNet-B3 ARCHITECTURE ----
model = efficientnet_b3(weights=None)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)

# ---- LOAD WEIGHTS FROM YOUR MODEL FILE ----
state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state["model_state_dict"])

model = model.to(device)
model.eval()

# ---- IMAGE PREPROCESSING ----
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict_image(image_file):
    image = Image.open(image_file).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)

    predicted_class = class_names[predicted.item()]
    confidence = torch.softmax(outputs, dim=1)[0][predicted.item()].item()

    return {
        "class": predicted_class,
        "confidence": float(confidence)
    }
