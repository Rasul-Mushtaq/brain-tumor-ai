# Detecting Brain Tumors Using AI

An end-to-end computer vision and full-stack web application that classifies brain tumor MRI scans into four categories (**Glioma**, **Meningioma**, **Pituitary**, and **No Tumor**) and visualizes how the neural network arrives at its decision across multiple convolutional depths using **Grad-CAM++**.

> _Note: This project is for learning purposes and not for diagnosis._

## Project Structure:

```txt
brain-tumor-ai/
├── ml_pipeline/
│   └── train.py                     # Trains model + auto-exports 4 sample MRI scans
├── backend/
│   ├── main.py                      # FastAPI + OOD entropy/confidence guardrail + Grad-CAM++
│   └── brain_tumor_resnet18.pth
└── frontend/
    ├── public/
    │   └── samples/                 # Auto-populated by train.py
    │       ├── glioma.jpg
    │       ├── meningioma.jpg
    │       ├── pituitary.jpg
    │       └── notumor.jpg
    └── src/
        ├── App.jsx                  # Editorial landing page + interactive XAI workbench
        └── App.css                  # Clean light-mode architectural typography & layout
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
uvicorn main:app --reload --port 8000
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

> _(Note: Because `.pth` model weight files are typically 40MB+ and are excluded in `.gitignore` to keep your repository clean, consider hosting your trained `brain_tumor_resnet18.pth` file under a **GitHub Release** on your repository so anyone cloning the project can download the weights directly without retraining.)_
