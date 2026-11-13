import React, { useState } from "react";
import "./App.css";

const PIPELINE_STEPS = [
  {
    id: "original",
    stepNumber: "01",
    title: "Preprocessed Input",
    layer: "224×224 Normalized Tensor",
    description:
      "The raw MRI scan is converted to a 3-channel tensor, resized to 224×224, and normalized using ImageNet channel statistics.",
  },
  {
    id: "layer2_heatmap",
    stepNumber: "02",
    title: "Early Edge & Texture Maps",
    layer: "ResNet18 · layer2[-1]",
    description:
      "Detects skull boundaries, tissue intensity gradients, and basic anatomical contours across the cranial cavity.",
  },
  {
    id: "layer3_heatmap",
    stepNumber: "03",
    title: "Mid-Level Structural Patterns",
    layer: "ResNet18 · layer3[-1]",
    description:
      "Fine-tuned convolutional blocks isolate abnormal tissue density, ventricular asymmetry, and mass boundaries.",
  },
  {
    id: "layer4_heatmap",
    stepNumber: "04",
    title: "Final Tumor Localization",
    layer: "ResNet18 · layer4[-1] (Grad-CAM++)",
    description:
      "High-level semantic activations pinpointing the exact region driving the final classification decision.",
  },
];

export default function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [activeStage, setActiveStage] = useState("layer4_heatmap");
  const [heatmapOpacity, setHeatmapOpacity] = useState(0.55);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setResult(null);
    setError("");
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setError("");

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch("http://localhost:8000/analyze", {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        throw new Error("Failed to analyze MRI scan.");
      }
      const data = await response.json();
      setResult(data);
      setActiveStage("layer4_heatmap");
    } catch (err) {
      setError(err.message || "Error connecting to inference server.");
    } finally {
      setLoading(false);
    }
  };

  const currentStepMeta =
    PIPELINE_STEPS.find((step) => step.id === activeStage) || PIPELINE_STEPS[3];

  return (
    <div className="app-container">
      <header className="top-bar">
        <div>
          <h1>NeuroScan XAI</h1>
          <p className="subtitle">
            Explainable Brain Tumor MRI Classification &amp; Grad-CAM++ Pipeline
          </p>
        </div>
        <span className="badge">ResNet18 · Two-Stage Fine-Tuned</span>
      </header>

      <main className="workspace">
        {/* Left Column: Upload & Pipeline Stepper */}
        <section className="panel control-panel">
          <h2>1. Upload MRI Scan</h2>
          <label className="upload-box">
            <input type="file" accept="image/*" onChange={handleFileChange} />
            {previewUrl ? (
              <img
                src={previewUrl}
                alt="Selected MRI"
                className="thumb-preview"
              />
            ) : (
              <span>Click to select a T1/T2 MRI scan (JPG, PNG)</span>
            )}
          </label>

          <button
            className="primary-btn"
            onClick={handleAnalyze}
            disabled={!selectedFile || loading}
          >
            {loading
              ? "Running Inference & Grad-CAM++..."
              : "Run Diagnostic Pipeline"}
          </button>

          {error && <div className="error-banner">{error}</div>}

          <div className="pipeline-section">
            <h2>2. Model Workflow Stages</h2>
            <p className="hint">
              Select a stage below to inspect how the network narrows its focus:
            </p>
            <div className="stepper-list">
              {PIPELINE_STEPS.map((step) => (
                <button
                  key={step.id}
                  className={`step-card ${activeStage === step.id ? "active" : ""}`}
                  onClick={() => setActiveStage(step.id)}
                  disabled={!result}
                >
                  <div className="step-header">
                    <span className="step-num">{step.stepNumber}</span>
                    <strong>{step.title}</strong>
                  </div>
                  <span className="step-layer">{step.layer}</span>
                </button>
              ))}
            </div>
          </div>
        </section>

        {/* Center Column: Interactive Heatmap Stage */}
        <section className="panel visualizer-panel">
          <div className="panel-header">
            <h2>Interactive Activation Inspector</h2>
            {result && activeStage !== "original" && (
              <div className="slider-control">
                <label htmlFor="opacity">
                  Heatmap Opacity: {Math.round(heatmapOpacity * 100)}%
                </label>
                <input
                  id="opacity"
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={heatmapOpacity}
                  onChange={(e) =>
                    setHeatmapOpacity(parseFloat(e.target.value))
                  }
                />
              </div>
            )}
          </div>

          <div className="canvas-wrapper">
            {!result ? (
              <div className="placeholder-stage">
                <p>
                  Upload an MRI scan and run the pipeline to inspect layer
                  activations.
                </p>
              </div>
            ) : (
              <div className="layered-viewport">
                <img
                  src={result.images.original}
                  alt="Base MRI"
                  className="base-mri"
                />
                {activeStage !== "original" && (
                  <img
                    src={result.images[activeStage]}
                    alt="Grad-CAM Overlay"
                    className="heatmap-overlay"
                    style={{ opacity: heatmapOpacity }}
                  />
                )}
              </div>
            )}
          </div>

          <div className="stage-explanation">
            <div className="stage-tag">{currentStepMeta.layer}</div>
            <h3>{currentStepMeta.title}</h3>
            <p>{currentStepMeta.description}</p>
          </div>
        </section>

        {/* Right Column: Classification Results & Confidence */}
        <section className="panel results-panel">
          <h2>3. Diagnostic Output</h2>
          {!result ? (
            <p className="hint">
              Results and class probability distributions will appear here.
            </p>
          ) : (
            <>
              <div className="prediction-card">
                <span className="pred-label">Primary Classification</span>
                <div className="pred-value">{result.prediction}</div>
                <div className="pred-conf">
                  Confidence: {result.confidence}%
                </div>
              </div>

              <h3>Class Probability Breakdown</h3>
              <div className="prob-list">
                {result.probabilities.map((item) => (
                  <div key={item.raw_label} className="prob-item">
                    <div className="prob-labels">
                      <span>{item.label}</span>
                      <span>{item.probability}%</span>
                    </div>
                    <div className="prob-bar-bg">
                      <div
                        className="prob-bar-fill"
                        style={{ width: `${item.probability}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>

              <div className="heatmap-legend">
                <span>Low Activation</span>
                <div className="jet-gradient" />
                <span>High Activation</span>
              </div>
            </>
          )}
        </section>
      </main>
    </div>
  );
}
