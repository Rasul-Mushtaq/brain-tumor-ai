# Detecting Brain Tumors Using AI

An end-to-end computer vision and full-stack web application that classifies brain tumor MRI scans into four categories (**Glioma**, **Meningioma**, **Pituitary**, and **No Tumor**) and visualizes how the neural network arrives at its decision across multiple convolutional depths using **Grad-CAM++**.

> _Note: This project is for learning purposes and not for diagnosis._

> ## Tech Stack

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)](https://reactjs.org/)
[![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)](https://developer.mozilla.org/en-US/docs/Web/JavaScript)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-38BDF8?style=for-the-badge&logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)
[![HTML](https://img.shields.io/badge/HTML-E34F26?style=for-the-badge&logo=html5&logoColor=white)](https://developer.mozilla.org/en-US/docs/Web/Guide/HTML/HTML5)

## Project Structure:

```txt
Directory structure:
└── rasul-mushtaq-brain-tumor-ai/
    ├── README.md
    ├── backend/
    │   ├── main.py               # FastAPI + OOD entropy/confidence guardrail + Grad-CAM++
    │   └── requirements.txt
    ├── frontend/
    │   ├── package.json
    │   ├── postcss.config.js
    │   ├── tailwind.config.js
    │   ├── public/
    │   │   └── index.html
    │   └── src/
    │       ├── App.jsx           # Landing page + interactive workbench
    │       ├── index.css
    │       ├── index.js
    │       └── styles/
    │           └── tailwind.css
    └── ml_pipeline/
        └── train.py              # Trains model + auto-exports 4 sample MRI scans
```

## Setup & Local Installation

**1. Train & Export the Model**

Run `ml_pipeline/train.py` in a GPU environment (such as Google Colab) to download the [Kaggle Brain Tumor MRI dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset), train the model, and generate `brain_tumor_resnet18.pth`:

```bash
cd ml_pipeline
pip install torch torchvision kagglehub scikit-learn
python train.py
```

Move the resulting `brain_tumor_resnet18.pth` file into the `backend/` directory.

**2. Start the FastAPI Backend**

```bash
cd backend
python -m venv .venv
source .venv\Scripts\activate        # On MacOS: source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Verify the API is active at `http://localhost:8000/health`.

**3. Start the React Frontend**

```bash
cd frontend
npm install
npm start
```

Open `http://localhost:3000` in your browser to upload MRI scans and inspect layer activations.

## API Reference

`POST /analyze`: Accepts a `multipart/form-data` image upload (`file`) and returns predicted class probabilities alongside Base64-encoded PNG visualizations:

```json
{
  "prediction": "Glioma",
  "confidence": 96.42,
  "probabilities": [
    { "label": "Glioma", "raw_label": "glioma", "probability": 96.42 },
    { "label": "Meningioma", "raw_label": "meningioma", "probability": 2.15 },
    { "label": "No Tumor", "raw_label": "notumor", "probability": 1.03 },
    { "label": "Pituitary", "raw_label": "pituitary", "probability": 0.4 }
  ],
  "images": {
    "original": "data:image/png;base64,...",
    "layer2_heatmap": "data:image/png;base64,...",
    "layer3_heatmap": "data:image/png;base64,...",
    "layer4_heatmap": "data:image/png;base64,..."
  }
}
```

---

### Data Source:

[Kaggle Brain Tumor MRI dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
