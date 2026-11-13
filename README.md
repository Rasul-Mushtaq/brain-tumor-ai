# Detecting Brain Tumors Using AI

> _Note: This project is for learning purposes and not for diagnosis._

## Project Structure:

```txt
brain-tumor-ai/
├── ml_pipeline/
│   ├── train.py               # Optimized two-stage training + metrics + model export
│   └── brain_tumor_resnet18.pth
├── backend/
│   ├── main.py                # FastAPI server with multi-layer Grad-CAM++ inference
│   └── requirements.txt
└── frontend/
    ├── package.json
    └── src/
        ├── App.jsx            # Main React dashboard with interactive heatmap inspector
        └── App.css            # Clean dark-mode clinical UI styling
```
