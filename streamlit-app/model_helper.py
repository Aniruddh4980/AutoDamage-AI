import os
import io
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

trained_model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class_names = ['F_Breakage', 'F_Crushed', 'F_Normal', 'R_Breakage', 'R_Crushed', 'R_Normal']

CLASS_METADATA = {
    'F_Breakage': {
        'display_name': 'Front Breakage',
        'location': 'Front End',
        'damage_type': 'Breakage / Crack',
        'severity': 'Severe',
        'color': '#ef4444',
        'description': 'Cracks, fractures, broken headlight/grille or split bumper detected at the front of the vehicle.'
    },
    'F_Crushed': {
        'display_name': 'Front Crushed',
        'location': 'Front End',
        'damage_type': 'Crushed / Structural Deformation',
        'severity': 'Moderate',
        'color': '#f59e0b',
        'description': 'Heavy frontal impact with hood crumple, bumper collapse, or radiator/chassis intrusion.'
    },
    'F_Normal': {
        'display_name': 'Front Normal',
        'location': 'Front End',
        'damage_type': 'No Damage Detected',
        'severity': 'None',
        'color': '#10b981',
        'description': 'Front body panels, hood, bumper, and headlights appear intact with no apparent structural damage.'
    },
    'R_Breakage': {
        'display_name': 'Rear Breakage',
        'location': 'Rear End',
        'damage_type': 'Breakage / Crack',
        'severity': 'Severe',
        'color': '#ef4444',
        'description': 'Cracked taillight, fractured rear bumper cover, or tailgate glass fracture detected.'
    },
    'R_Crushed': {
        'display_name': 'Rear Crushed',
        'location': 'Rear End',
        'damage_type': 'Crushed / Structural Deformation',
        'severity': 'Moderate',
        'color': '#f59e0b',
        'description': 'Severe rear-end collision damage with trunk intrusion, bumper displacement, or rear quarter impact.'
    },
    'R_Normal': {
        'display_name': 'Rear Normal',
        'location': 'Rear End',
        'damage_type': 'No Damage Detected',
        'severity': 'None',
        'color': '#10b981',
        'description': 'Rear bumper, trunk lid, and taillights appear intact with no visible structural deformation.'
    }
}

class CarClassifierResNet(nn.Module):
    def __init__(self, num_classes=6, dropout_rate=0.44):
        super().__init__()
        self.model = models.resnet50(weights='DEFAULT')
        # Freeze all layers except the final fully connected layer
        for param in self.model.parameters():
            param.requires_grad = False
        # Unfreeze layer 4 and fc layers
        for param in self.model.layer4.parameters():
            param.requires_grad = True
        # Replace the final fully connected layer
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_model():
    global trained_model
    if trained_model is None:
        model = CarClassifierResNet()
        model_path = os.path.join(os.path.dirname(__file__), 'model', 'saved_model.pth')
        if not os.path.exists(model_path):
            # Fallback if run from repo root
            model_path = os.path.join('model', 'saved_model.pth')
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        trained_model = model
    return trained_model

def _load_image(image_input):
    """Loads image from path, bytes, buffer, or PIL Image."""
    if isinstance(image_input, Image.Image):
        return image_input.convert('RGB')
    if isinstance(image_input, (str, os.PathLike)):
        return Image.open(image_input).convert('RGB')
    if isinstance(image_input, (bytes, bytearray)):
        return Image.open(io.BytesIO(image_input)).convert('RGB')
    if hasattr(image_input, 'read'):
        image_input.seek(0)
        return Image.open(image_input).convert('RGB')
    raise ValueError(f"Unsupported image input type: {type(image_input)}")

def predict(image_path):
    """Backward-compatible predict function returning class name."""
    res = predict_detailed(image_path)
    return res['class']

def predict_detailed(image_input):
    """Comprehensive prediction returning softmax probabilities and metadata."""
    image = _load_image(image_input)
    transform = get_transform()
    image_tensor = transform(image).unsqueeze(0).to(device)

    model = get_model()

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, dim=0)
        
        predicted_class = class_names[predicted_idx.item()]
        conf_val = float(confidence.item())

        prob_dict = {
            c_name: float(probabilities[i].item())
            for i, c_name in enumerate(class_names)
        }

    meta = CLASS_METADATA.get(predicted_class, {
        'display_name': predicted_class,
        'location': 'Front' if predicted_class.startswith('F_') else 'Rear',
        'damage_type': 'Damage' if 'Normal' not in predicted_class else 'Normal',
        'severity': 'Moderate',
        'color': '#3b82f6',
        'description': ''
    })

    return {
        'class': predicted_class,
        'display_name': meta['display_name'],
        'location': meta['location'],
        'damage_type': meta['damage_type'],
        'severity': meta['severity'],
        'color': meta['color'],
        'description': meta['description'],
        'confidence': conf_val,
        'probabilities': prob_dict,
        'image_size': image.size,
        'device': str(device)
    }