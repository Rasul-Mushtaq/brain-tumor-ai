import io
import base64
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

app = FastAPI(title="NeuroScan XAI Inference Engine")

# Allow requests from your React development server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model checkpoint
try:
    checkpoint = torch.load("brain_tumor_resnet18.pth", map_location=device)
    CLASS_NAMES = checkpoint["class_names"]
    MODEL_ACCURACY = checkpoint.get("accuracy", 0.0)
except Exception as e:
    raise RuntimeError(f"Failed to load 'brain_tumor_resnet18.pth'. Did you run train.py first? Error: {e}")

model = models.resnet18(weights=None)
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.fc.in_features, len(CLASS_NAMES))
)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

for param in model.parameters():
    param.requires_grad = True

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

CONFIDENCE_THRESHOLD = 0.65
ENTROPY_THRESHOLD = 0.72

def encode_rgb_to_base64(rgb_array: np.ndarray) -> str:
    pil_img = Image.fromarray(rgb_array.astype(np.uint8))
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

def generate_layer_cam(input_tensor: torch.Tensor, target_layer, target_class_idx: int) -> np.ndarray:
    targets = [ClassifierOutputTarget(target_class_idx)]
    with GradCAMPlusPlus(model=model, target_layers=[target_layer]) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    heatmap_uint8 = np.uint8(255 * grayscale_cam)
    heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

def compute_normalized_entropy(probs: np.ndarray) -> float:
    eps = 1e-12
    entropy = -np.sum(probs * np.log(probs + eps))
    max_entropy = np.log(len(probs))
    return float(entropy / max_entropy)

@app.get("/health")
def health_check():
    return {
        "status": "online",
        "classes": CLASS_NAMES,
        "test_accuracy": round(MODEL_ACCURACY * 100, 2)
    }

@app.post("/analyze")
async def analyze_mri(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a valid image.")

    contents = await file.read()
    try:
        raw_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Unable to decode uploaded image file.")

    resized_rgb = np.array(raw_image.resize((224, 224)))
    input_tensor = preprocess(raw_image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])
    norm_entropy = compute_normalized_entropy(probs)

    is_ood = bool(confidence < CONFIDENCE_THRESHOLD or norm_entropy > ENTROPY_THRESHOLD)

    # Generate Grad-CAM++ heatmaps safely across network layers
    cam_early = generate_layer_cam(input_tensor, model.layer2[-1], pred_idx)
    cam_mid = generate_layer_cam(input_tensor, model.layer3[-1], pred_idx)
    cam_final = generate_layer_cam(input_tensor, model.layer4[-1], pred_idx)

    probabilities = [
        {
            "label": CLASS_NAMES[i].replace("_", " ").title(),
            "raw_label": CLASS_NAMES[i],
            "probability": round(float(probs[i]) * 100, 2)
        }
        for i in range(len(CLASS_NAMES))
    ]
    probabilities.sort(key=lambda item: item["probability"], reverse=True)

    return {
        "prediction": "Uncertain / Non-MRI Input" if is_ood else pred_class.replace("_", " ").title(),
        "raw_prediction": pred_class.replace("_", " ").title(),
        "confidence": round(confidence * 100, 2),
        "entropy": round(norm_entropy, 3),
        "is_ood": is_ood,
        "probabilities": probabilities,
        "images": {
            "original": encode_rgb_to_base64(resized_rgb),
            "layer2_heatmap": encode_rgb_to_base64(cam_early),
            "layer3_heatmap": encode_rgb_to_base64(cam_mid),
            "layer4_heatmap": encode_rgb_to_base64(cam_final),
        }
    }