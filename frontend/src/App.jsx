import React, { useState, useRef } from "react";
import "./App.css";

const SAMPLE_SCANS = [
  {
    id: "glioma",
    title: "Glioma",
    note: "Intra-axial infiltrative mass",
    src: "/samples/glioma.jpg",
  },
  {
    id: "meningioma",
    title: "Meningioma",
    note: "Extra-axial dural attachment",
    src: "/samples/meningioma.jpg",
  },
  {
    id: "pituitary",
    title: "Pituitary",
    note: "Sellar region adenoma",
    src: "/samples/pituitary.jpg",
  },
  {
    id: "notumor",
    title: "No Tumor",
    note: "Normal cranial parenchyma",
    src: "/samples/notumor.jpg",
  },
];

const PIPELINE_STAGES = [
  {
    id: "original",
    index: "01",
    name: "Input Normalization",
    module: "224x224 RGB Tensor",
    summary:
      "The raw scan is standardized to 224x224 resolution, converted to three channels, and normalized against training distribution statistics.",
  },
  {
    id: "layer2_heatmap",
    index: "02",
    name: "Cranial Edge Extraction",
    module: "ResNet18 layer2[-1]",
    summary:
      "Early residual blocks isolate skull boundaries, cortical folding, and high-contrast tissue interfaces across the scan.",
  },
  {
    id: "layer3_heatmap",
    index: "03",
    name: "Structural Asymmetry",
    module: "ResNet18 layer3[-1]",
    summary:
      "Intermediate convolutional filters respond to ventricular displacement, mass effect, and abnormal intensity gradients.",
  },
  {
    id: "layer4_heatmap",
    index: "04",
    name: "Pathology Localization",
    module: "ResNet18 layer4[-1] Grad-CAM++",
    summary:
      "Second-order gradient weighting isolates the specific anatomical region responsible for the final classification output.",
  },
];

export default function App() {
  const [selectedSampleId, setSelectedSampleId] = useState(null);
  const [fileName, setFileName] = useState("");
  const [result, setResult] = useState(null);
  const [activeStage, setActiveStage] = useState("layer4_heatmap");
  const [opacity, setOpacity] = useState(0.55);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const workbenchRef = useRef(null);
  const fileInputRef = useRef(null);

  const scrollToWorkbench = () => {
    workbenchRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const runInference = async (fileBlob, displayName) => {
    setLoading(true);
    setError("");
    setFileName(displayName);

    const formData = new FormData();
    formData.append("file", fileBlob, displayName);

    try {
      const response = await fetch("http://localhost:8000/analyze", {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || "Inference request failed.");
      }
      const data = await response.json();
      setResult(data);
      setActiveStage("layer4_heatmap");
    } catch (err) {
      setError(err.message || "Unable to reach the inference server.");
    } finally {
      setLoading(false);
    }
  };

  const handleSampleSelect = async (sample) => {
    setSelectedSampleId(sample.id);
    try {
      const res = await fetch(sample.src);
      if (!res.ok) throw new Error("Sample scan not found in public/samples/.");
      const blob = await res.blob();
      await runInference(blob, `${sample.id}.jpg`);
    } catch (err) {
      setError(err.message);
    }
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setSelectedSampleId(null);
    await runInference(file, file.name);
  };

  const currentStage =
    PIPELINE_STAGES.find((stage) => stage.id === activeStage) ||
    PIPELINE_STAGES[3];

  return (
    <div className="site-shell">
      <nav className="top-nav">
        <a href="#top" className="brand-mark">
          Brain Tumor AI Detector
        </a>
        <div className="nav-links">
          <a href="#methodology">Methodology</a>
          <a href="#workbench">Diagnostic Workbench</a>
        </div>
      </nav>

      {/* Editorial Hero */}
      <header id="top" className="hero-section">
        <div className="hero-grid">
          <div className="hero-copy">
            <h1>
              Interpretable cranial MRI classification with multi-layer
              Grad-CAM++.
            </h1>
            <p className="hero-lead">
              A fine-tuned ResNet18 vision pipeline that classifies glioma,
              meningioma, pituitary tumors, and healthy scans while exposing
              spatial feature activations across network depths.
            </p>
            <div className="hero-actions">
              <button className="btn-solid" onClick={scrollToWorkbench}>
                Launch Interactive Workbench
              </button>
              <a href="#methodology" className="btn-outline">
                Review Architecture
              </a>
            </div>
          </div>

          <div className="hero-metrics">
            <div className="metric-block">
              <span className="metric-number">4</span>
              <span className="metric-caption">
                Diagnostic classes evaluated across 1,600 unseen test scans
              </span>
            </div>
            <div className="metric-block">
              <span className="metric-number">3</span>
              <span className="metric-caption">
                Convolutional depths mapped from early edges to tumor
                localization
              </span>
            </div>
            <div className="metric-block">
              <span className="metric-number">65%</span>
              <span className="metric-caption">
                Confidence floor paired with Shannon entropy filtering for
                out-of-distribution inputs
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* Architecture Overview */}
      <section id="methodology" className="methodology-section">
        <div className="section-heading">
          <h2>Pipeline Architecture</h2>
          <p>
            Medical imaging requires both spatial precision and calibrated
            uncertainty. Each scan passes through three distinct stages before a
            classification is surfaced.
          </p>
        </div>

        <div className="method-columns">
          <article className="method-item">
            <h3>Two-Stage Fine-Tuning</h3>
            <p>
              The classification head is trained first to stabilize gradients
              before unfreezing residual blocks layer3 and layer4 under a cosine
              annealing schedule, adapting pretrained filters to cranial tissue
              contrast.
            </p>
          </article>
          <article className="method-item">
            <h3>Second-Order Localization</h3>
            <p>
              Grad-CAM++ computes pixel-wise gradient weights across multiple
              convolutional layers, producing sharper boundaries on irregular
              lesions than standard global average pooling.
            </p>
          </article>
          <article className="method-item">
            <h3>Out-of-Distribution Guardrail</h3>
            <p>
              Predictions falling below 65% confidence or exceeding 0.72
              normalized prediction entropy are flagged as uncertain, preventing
              forced classifications on non-MRI uploads.
            </p>
          </article>
        </div>
      </section>

      {/* Interactive Diagnostic Workbench */}
      <section id="workbench" ref={workbenchRef} className="workbench-section">
        <div className="workbench-header">
          <div>
            <h2>Diagnostic Workbench</h2>
            <p>
              Select a reference MRI scan below or upload a local image to
              inspect layer activations.
            </p>
          </div>

          <div className="upload-trigger">
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileUpload}
              hidden
            />
            <button
              className="btn-outline"
              onClick={() => fileInputRef.current?.click()}
              disabled={loading}
            >
              Upload Custom Scan
            </button>
          </div>
        </div>

        {/* One-Click Sample Gallery */}
        <div className="sample-strip">
          {SAMPLE_SCANS.map((sample) => (
            <button
              key={sample.id}
              className={`sample-card ${selectedSampleId === sample.id ? "selected" : ""}`}
              onClick={() => handleSampleSelect(sample)}
              disabled={loading}
            >
              <img
                src={sample.src}
                alt={sample.title}
                className="sample-thumb"
              />
              <div className="sample-meta">
                <strong>{sample.title}</strong>
                <span>{sample.note}</span>
              </div>
            </button>
          ))}
        </div>

        {error && <div className="status-alert error">{error}</div>}

        {/* Main Inspector Grid */}
        <div className="inspector-grid">
          {/* Left: Layer Depth Selector */}
          <div className="inspector-column stages-col">
            <h3>Activation Depth</h3>
            <div className="stage-list">
              {PIPELINE_STAGES.map((stage) => (
                <button
                  key={stage.id}
                  className={`stage-row ${activeStage === stage.id ? "active" : ""}`}
                  onClick={() => setActiveStage(stage.id)}
                  disabled={!result || loading}
                >
                  <span className="stage-idx">{stage.index}</span>
                  <div className="stage-text">
                    <strong>{stage.name}</strong>
                    <span>{stage.module}</span>
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Center: Layered Scan Viewport */}
          <div className="inspector-column viewport-col">
            <div className="viewport-toolbar">
              <span className="active-file-label">
                {loading
                  ? "Processing scan..."
                  : fileName
                    ? `Active scan: ${fileName}`
                    : "No scan loaded"}
              </span>

              {result && activeStage !== "original" && (
                <div className="opacity-control">
                  <label htmlFor="cam-opacity">
                    Overlay {Math.round(opacity * 100)}%
                  </label>
                  <input
                    id="cam-opacity"
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={opacity}
                    onChange={(e) => setOpacity(parseFloat(e.target.value))}
                  />
                </div>
              )}
            </div>

            <div className="scan-stage">
              {!result ? (
                <div className="empty-stage">
                  <p>
                    Choose any sample scan above or upload an image to begin
                    inspection.
                  </p>
                </div>
              ) : (
                <div className="scan-composite">
                  <img
                    src={result.images.original}
                    alt="Normalized MRI"
                    className="scan-base"
                  />
                  {activeStage !== "original" && (
                    <img
                      src={result.images[activeStage]}
                      alt="Grad-CAM++ Activation"
                      className="scan-heatmap"
                      style={{ opacity }}
                    />
                  )}
                </div>
              )}
            </div>

            <div className="stage-caption">
              <strong>{currentStage.name}</strong>
              <p>{currentStage.summary}</p>
            </div>
          </div>

          {/* Right: Diagnostic Output & OOD Guardrail */}
          <div className="inspector-column output-col">
            <h3>Classification Output</h3>

            {!result ? (
              <p className="muted-note">
                Probability distributions and confidence diagnostics will appear
                here once a scan is analyzed.
              </p>
            ) : (
              <>
                {result.is_ood && (
                  <div className="ood-warning">
                    <strong>Out-of-Distribution Input Detected</strong>
                    <p>
                      Model confidence ({result.confidence}%) or prediction
                      entropy ({result.entropy}) indicates this image may not be
                      a valid cranial MRI scan.
                    </p>
                  </div>
                )}

                <div
                  className={`diagnosis-summary ${result.is_ood ? "uncertain" : ""}`}
                >
                  <span className="summary-label">Primary Prediction</span>
                  <div className="summary-title">{result.prediction}</div>
                  <div className="summary-metrics">
                    <span>Confidence: {result.confidence}%</span>
                    <span>Entropy: {result.entropy}</span>
                  </div>
                </div>

                <div className="distribution-block">
                  <h4>Class Distribution</h4>
                  {result.probabilities.map((item) => (
                    <div key={item.raw_label} className="dist-row">
                      <div className="dist-header">
                        <span>{item.label}</span>
                        <span>{item.probability}%</span>
                      </div>
                      <div className="dist-track">
                        <div
                          className="dist-fill"
                          style={{ width: `${item.probability}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>

                <div className="scale-bar">
                  <span>Low activation</span>
                  <div className="scale-gradient" />
                  <span>High activation</span>
                </div>
              </>
            )}
          </div>
        </div>
      </section>

      {/* Medical Disclaimer Footer */}
      <footer className="site-footer">
        <p>
          Medical Disclaimer: This tool is an educational demonstration
          developed for research and studying. It is not a cleared medical
          device and must not be used for clinical diagnosis, triage, or patient
          treatment decisions.
        </p>
      </footer>
    </div>
  );
}
