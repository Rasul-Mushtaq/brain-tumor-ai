import os
import io
import base64
from contextlib import asynccontextmanager
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Global variables
model = None
CLASS_NAMES = []
MODEL_ACCURACY = 0.0
device = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, CLASS_NAMES, MODEL_ACCURACY, device
    
    print("Initializing environment and PyTorch...")
    import torch
    import torch.nn as nn
    from torchvision import models
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading model checkpoint...")
    checkpoint = torch.load("brain_tumor_resnet18.pth", map_location="cpu")
    CLASS_NAMES = checkpoint["class_names"]
    MODEL_ACCURACY = checkpoint.get("accuracy", 0.0)

    temp_model = models.resnet18(weights=None)
    temp_model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(temp_model.fc.in_features, len(CLASS_NAMES))
    )
    temp_model.load_state_dict(checkpoint["model_state_dict"])
    temp_model.to(device)
    temp_model.eval()
    
    model = temp_model
    print("Startup complete!")
    yield
    model = None

app = FastAPI(title="Brain Tumor MRI Scans Inference Engine", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {
        "status": "online",
        "classes": CLASS_NAMES,
        "test_accuracy": round(MODEL_ACCURACY * 100, 2)
    }

@app.post("/analyze")
async def analyze_mri(file: UploadFile = File(...)):
    import torch
    import torch.nn.functional as F
    from torchvision import transforms
    import cv2
    from pytorch_grad_cam import GradCAMPlusPlus
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a valid image.")

    contents = await file.read()
    try:
        raw_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Unable to decode uploaded image file.")

    resized_rgb = np.array(raw_image.resize((224, 224)))
    
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    preprocess = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std)
    ])
    
    input_tensor = preprocess(raw_image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])
    
    # Entropy calculation
    eps = 1e-12
    entropy = -np.sum(probs * np.log(probs + eps))
    norm_entropy = float(entropy / np.log(len(probs)))
    is_ood = bool(confidence < 0.65 or norm_entropy > 0.72)

    # Grad-CAM++ generation
    def generate_cam(target_layer):
        targets = [ClassifierOutputTarget(pred_idx)]
        with GradCAMPlusPlus(model=model, target_layers=[target_layer]) as cam:
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
        heatmap = np.uint8(255 * grayscale_cam)
        heatmap_bgr = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    cam_early = generate_cam(model.layer2[-1])
    cam_mid = generate_cam(model.layer3[-1])
    cam_final = generate_cam(model.layer4[-1])

    def encode_rgb(rgb):
        pil_img = Image.fromarray(rgb.astype(np.uint8))
        buffer = io.BytesIO()
        pil_img.save(buffer, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"

    probabilities = [
        {
            "label": CLASS_NAMES[i].replace("_", " ").title(),
            "raw_label": CLASS_NAMES[i],
            "probability": round(float(probs[i]) * 100, 2)
        }
        for i in range(len(CLASS_NAMES))
    ]
    probabilities.sort(key=lambda x: x["probability"], reverse=True)

    return {
        "prediction": "Uncertain / Non-MRI Input" if is_ood else pred_class.replace("_", " ").title(),
        "confidence": round(confidence * 100, 2),
        "entropy": round(norm_entropy, 3),
        "is_ood": is_ood,
        "probabilities": probabilities,
        "images": {
            "original": encode_rgb(resized_rgb),
            "layer2_heatmap": encode_rgb(cam_early),
            "layer3_heatmap": encode_rgb(cam_mid),
            "layer4_heatmap": encode_rgb(cam_final),
        }
    }