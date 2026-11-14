import React, { useState, useRef } from "react";

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

const ARCHITECTURE_STEPS = [
  {
    step: "01",
    title: "Two-Stage Fine-Tuning",
    accent: "bg-sky-500",
    description:
      "The classification head is trained first to stabilize gradients before unfreezing residual blocks layer3 and layer4 under a cosine annealing schedule, adapting pretrained filters to cranial tissue contrast.",
  },
  {
    step: "02",
    title: "Second-Order Localization",
    accent: "bg-orange-600",
    description:
      "Grad-CAM++ computes pixel-wise gradient weights across multiple convolutional layers, producing sharper boundaries on irregular lesions than standard global average pooling.",
  },
  {
    step: "03",
    title: "Out-of-Distribution Guardrail",
    accent: "bg-[#0c1e35]",
    description:
      "Predictions falling below 65% confidence or exceeding 0.72 normalized prediction entropy are flagged as uncertain, preventing forced classifications on non-MRI uploads.",
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

  const [inspectMedia, setInspectMedia] = useState(null);
  const [inspectZoom, setInspectZoom] = useState(1);

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
      const response = await fetch("http://127.0.0.1:8000/analyze", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `Server error: ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
      setActiveStage("layer4_heatmap");
    } catch (err) {
      setError(
        err.message.includes("Failed to fetch")
          ? "Unable to connect to the backend server. Please verify FastAPI is running on http://127.0.0.1:8000."
          : err.message,
      );
    } finally {
      setLoading(false);
    }
  };

  const handleSampleSelect = async (sample) => {
    setSelectedSampleId(sample.id);
    try {
      const res = await fetch(sample.src);
      if (!res.ok)
        throw new Error(
          `Sample scan '${sample.src}' not found in public/samples/.`,
        );
      const blob = await res.blob();
      await runInference(blob, `${sample.id}.jpg`);
    } catch (err) {
      setError(err.message);
      setLoading(false);
    }
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setSelectedSampleId(null);
    await runInference(file, file.name);
  };

  const openInspectMode = (mediaObj) => {
    setInspectZoom(1);
    setInspectMedia(mediaObj);
  };

  const currentStage =
    PIPELINE_STAGES.find((stage) => stage.id === activeStage) ||
    PIPELINE_STAGES[3];

  return (
    <div className="min-h-screen bg-[#f5f7fa] text-[#0c1e35] font-sans antialiased selection:bg-sky-500 selection:text-white">
      {/* Navigation */}
      <header className="sticky top-0 z-30 w-full border-b border-slate-200 bg-white/95 backdrop-blur-md shadow-xs">
        <nav className="max-w-7xl mx-auto px-8 h-20 flex justify-between items-center">
          <a
            href="#top"
            className="text-lg font-bold tracking-tight text-[#0c1e35] flex items-center"
          >
            Classifying Brain Tumors
          </a>

          <div className="flex items-center gap-8 text-sm font-medium text-slate-700">
            <a
              href="#methodology"
              className="hover:text-sky-600 transition-colors"
            >
              Methodology
            </a>
            <a
              href="#workbench"
              className="hover:text-sky-600 transition-colors"
            >
              Diagnostic Workbench
            </a>
            <a
              href="https://github.com/Rasul-Mushtaq/brain-tumor-ai"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 px-4 py-2 rounded-lg border border-slate-300 bg-slate-50 text-slate-800 hover:bg-slate-100 hover:text-[#0c1e35] hover:border-slate-400 transition-all text-sm font-medium shadow-2xs"
            >
              <svg
                className="w-5 h-5 fill-current"
                viewBox="0 0 24 24"
                aria-hidden="true"
              >
                <path
                  fillRule="evenodd"
                  clipRule="evenodd"
                  d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.53 1.032 1.53 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z"
                />
              </svg>
              <span>GitHub Repository</span>
            </a>
          </div>
        </nav>
      </header>

      {/* Hero Section */}
      <header
        id="top"
        className="max-w-7xl mx-auto px-8 py-20 lg:py-28 border-b border-slate-200"
      >
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
          <div>
            <h1 className="text-4xl lg:text-5xl font-extrabold tracking-tight text-[#0c1e35] leading-[1.12] mb-6">
              Interpretable cranial MRI classification with multi-layer
              Grad-CAM++.
            </h1>
            <p className="text-lg text-slate-600 leading-relaxed max-w-xl mb-10">
              A fine-tuned ResNet18 vision pipeline that classifies glioma,
              meningioma, pituitary tumors, and healthy scans while exposing
              spatial feature activations across network depths.
            </p>
            <div className="flex gap-4 flex-wrap">
              <button
                onClick={scrollToWorkbench}
                className="bg-sky-600 hover:bg-sky-700 text-white font-semibold px-7 py-4 rounded-xl shadow-sm transition-all hover:-translate-y-0.5"
              >
                Start Interactive Workbench
              </button>
              <a
                href="#methodology"
                className="bg-white hover:bg-sky-50/50 text-slate-900 border border-slate-300 hover:border-sky-500 font-medium px-7 py-4 rounded-xl transition-all"
              >
                Review Pipeline
              </a>
            </div>
          </div>

          {/* Asymmetric Showcase with Stable Hover & Click-to-Inspect */}
          <div className="bg-white border border-slate-200 border-t-4 border-t-sky-500 rounded-2xl p-8 shadow-xl">
            <div className="flex flex-col gap-6">
              <figure
                className="bg-[#070d17] border border-slate-200 rounded-xl p-4 cursor-zoom-in transition-all duration-300 hover:scale-[1.02] hover:border-sky-500 hover:shadow-lg w-[85%] self-start group"
                onClick={() =>
                  openInspectMode({
                    baseSrc: "/brain_tumor_sample.jpg",
                    title: "Raw Cranial MRI Input",
                    detail:
                      "Unprocessed T1-weighted cranial scan prior to tensor normalization.",
                  })
                }
              >
                <img
                  src="/brain_tumor_sample.jpg"
                  alt="Raw Brain Tumor MRI Scan"
                  className="w-full max-h-[220px] object-contain block rounded"
                />
                <figcaption className="w-full mt-3 text-xs text-slate-300 flex justify-between items-center">
                  <span className="font-medium text-slate-200">
                    Input MRI Scan
                  </span>
                  <span className="text-sky-400 group-hover:underline text-[11px]">
                    Click to inspect →
                  </span>
                </figcaption>
              </figure>

              <figure
                className="bg-[#070d17] border border-sky-900/30 rounded-xl p-4 cursor-zoom-in transition-all duration-300 hover:scale-[1.02] hover:border-sky-500 hover:shadow-lg w-[90%] self-end group"
                onClick={() =>
                  openInspectMode({
                    baseSrc: "/brain_tumor_heatmap.jpg",
                    title: "Grad-CAM++ Spatial Localization",
                    detail:
                      "Second-order gradient activation map highlighting neoplastic tissue.",
                  })
                }
              >
                <img
                  src="/brain_tumor_heatmap.jpg"
                  alt="Grad-CAM++ Heatmap Activation"
                  className="w-full max-h-[220px] object-contain block rounded"
                />
                <figcaption className="w-full mt-3 text-xs text-slate-300 flex justify-between items-center">
                  <span className="font-medium text-slate-200">
                    Grad-CAM++ Localization
                  </span>
                  <span className="text-sky-400 group-hover:underline text-[11px]">
                    Click to inspect →
                  </span>
                </figcaption>
              </figure>
            </div>
            <p className="mt-6 pt-4 border-t border-slate-100 text-xs text-slate-500 text-center font-medium">
              Click any image to open full-screen interactive inspection mode.
            </p>
          </div>
        </div>
      </header>

      {/* Methodology Section (Redesigned Editorial Layout) */}
      <section
        id="methodology"
        className="max-w-7xl mx-auto px-8 py-20 lg:py-28 border-b border-slate-200"
      >
        <div className="max-w-2xl mb-14">
          <h2 className="text-3xl font-extrabold tracking-tight text-[#0c1e35] mb-4">
            Pipeline Architecture
          </h2>
          <p className="text-slate-600 text-base leading-relaxed">
            Medical imaging requires both spatial precision and calibrated
            uncertainty. Each scan passes through three structured phases before
            surfacing a final classification.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {ARCHITECTURE_STEPS.map((item, idx) => (
            <div
              key={idx}
              className="bg-white border border-slate-200 rounded-2xl p-8 shadow-sm flex flex-col justify-between relative overflow-hidden group hover:border-sky-400 transition-colors"
            >
              <div className="absolute top-0 right-0 p-6 text-5xl font-black text-slate-100 select-none group-hover:text-sky-50/80 transition-colors">
                {item.step}
              </div>
              <div>
                <div
                  className={`w-10 h-1.5 ${item.accent} rounded-full mb-6`}
                ></div>
                <h3 className="text-xl font-bold text-[#0c1e35] mb-3">
                  {item.title}
                </h3>
                <p className="text-slate-600 text-sm leading-relaxed">
                  {item.description}
                </p>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Diagnostic Workbench Section */}
      <section
        id="workbench"
        ref={workbenchRef}
        className="max-w-7xl mx-auto px-8 py-20 lg:py-28"
      >
        <div className="flex justify-between items-end gap-6 mb-10 flex-wrap">
          <div>
            <h2 className="text-3xl font-extrabold tracking-tight text-[#0c1e35] mb-2">
              Diagnostic Workbench
            </h2>
            <p className="text-slate-600 text-base">
              Select a reference MRI scan below or upload a local image to
              inspect layer activations.
            </p>
          </div>
          <div>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileUpload}
              className="hidden"
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={loading}
              className="bg-[#0c1e35] hover:bg-[#163254] text-white font-semibold px-6 py-3.5 rounded-xl text-sm transition-colors shadow-sm disabled:opacity-50"
            >
              Upload Custom Scan
            </button>
          </div>
        </div>

        {/* Sample Gallery Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 mb-10">
          {SAMPLE_SCANS.map((sample) => (
            <button
              key={sample.id}
              onClick={() => handleSampleSelect(sample)}
              disabled={loading}
              className={`flex flex-col bg-white border rounded-2xl p-5 text-left transition-all shadow-sm hover:-translate-y-1 hover:shadow-md ${selectedSampleId === sample.id ? "border-sky-500 border-2 bg-sky-50/40 shadow-sm" : "border-slate-200"}`}
            >
              <div className="w-full h-48 bg-[#070d17] rounded-xl overflow-hidden flex items-center justify-center mb-4 border border-slate-100">
                <img
                  src={sample.src}
                  alt={sample.title}
                  className="w-full h-full object-contain"
                />
              </div>
              <strong className="text-base font-bold text-[#0c1e35] mb-1">
                {sample.title}
              </strong>
              <span className="text-xs text-slate-500 leading-normal">
                {sample.note}
              </span>
            </button>
          ))}
        </div>

        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 px-5 py-4 rounded-xl text-sm mb-8">
            {error}
          </div>
        )}

        {/* Three Column Inspector */}
        <div className="grid grid-cols-1 lg:grid-cols-[310px_1fr_330px] gap-8 items-stretch">
          {/* Stage Selector */}
          <div className="bg-white border border-slate-200 rounded-2xl p-7 flex flex-col shadow-sm">
            <h3 className="text-sm font-bold uppercase tracking-wider text-slate-400 mb-5">
              Activation Depth
            </h3>
            <div className="flex flex-col gap-3">
              {PIPELINE_STAGES.map((stage) => (
                <button
                  key={stage.id}
                  onClick={() => setActiveStage(stage.id)}
                  disabled={!result || loading}
                  className={`flex items-start gap-3.5 p-4 bg-slate-50/60 border rounded-xl text-left transition-all ${activeStage === stage.id ? "border-sky-500 bg-sky-50/50 shadow-sm" : "border-slate-200 hover:border-slate-300"} disabled:opacity-50`}
                >
                  <span
                    className={`text-xs font-extrabold pt-0.5 ${activeStage === stage.id ? "text-sky-600" : "text-slate-400"}`}
                  >
                    {stage.index}
                  </span>
                  <div className="flex flex-col">
                    <strong className="text-sm font-bold text-[#0c1e35] mb-0.5">
                      {stage.name}
                    </strong>
                    <span className="text-xs text-slate-500">
                      {stage.module}
                    </span>
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Viewport Canvas */}
          <div className="bg-white border border-slate-200 rounded-2xl p-7 flex flex-col shadow-sm">
            <div className="flex justify-between items-center mb-5 gap-4 min-h-[32px] flex-wrap">
              <span className="text-xs font-semibold text-slate-600">
                {loading
                  ? "Processing scan through inference engine..."
                  : fileName
                    ? `Active scan: ${fileName}`
                    : "No scan loaded"}
              </span>
              {result && activeStage !== "original" && (
                <div className="flex items-center gap-3 text-xs font-medium text-slate-600 bg-slate-50 px-3 py-1.5 rounded-lg border border-slate-200">
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
                    className="accent-sky-500 cursor-pointer"
                  />
                </div>
              )}
            </div>

            <div className="bg-[#070d17] rounded-xl min-h-[420px] flex items-center justify-center p-8 mb-6 border border-slate-900">
              {!result ? (
                <div className="text-slate-400 text-sm text-center max-w-[280px] leading-relaxed">
                  Choose any sample scan above or upload an image to begin
                  inspection.
                </div>
              ) : (
                <div
                  className="relative inline-flex items-center justify-center max-w-full max-h-[380px] rounded-lg overflow-hidden cursor-zoom-in border border-white/10 transition-transform hover:scale-[1.02] shadow-xl group"
                  onClick={() =>
                    openInspectMode({
                      baseSrc: result.images.original,
                      overlaySrc:
                        activeStage !== "original"
                          ? result.images[activeStage]
                          : null,
                      overlayOpacity: opacity,
                      title: `${currentStage.name} (${fileName})`,
                      detail: currentStage.summary,
                    })
                  }
                >
                  <img
                    src={result.images.original}
                    alt="Normalized MRI"
                    className="max-w-full max-h-[380px] w-auto h-auto object-contain block"
                  />
                  {activeStage !== "original" && (
                    <img
                      src={result.images[activeStage]}
                      alt="Grad-CAM++ Activation"
                      className="absolute inset-0 w-full h-full object-contain mix-blend-screen transition-opacity"
                      style={{ opacity }}
                    />
                  )}
                  <div className="absolute bottom-3 right-3 bg-black/80 backdrop-blur-md text-white text-[11px] px-2.5 py-1 rounded-md opacity-0 group-hover:opacity-100 transition-opacity">
                    Click to inspect $\rightarrow$
                  </div>
                </div>
              )}
            </div>

            <div className="border-t border-slate-200 pt-5 mt-auto">
              <strong className="block text-sm font-bold text-[#0c1e35] mb-1.5">
                {currentStage.name}
              </strong>
              <p className="text-xs text-slate-600 leading-relaxed">
                {currentStage.summary}
              </p>
            </div>
          </div>

          {/* Diagnostic Output */}
          <div className="bg-white border border-slate-200 rounded-2xl p-7 flex flex-col shadow-sm">
            <h3 className="text-sm font-bold uppercase tracking-wider text-slate-400 mb-5">
              Classification Output
            </h3>
            {!result ? (
              <p className="text-slate-400 text-sm leading-relaxed">
                Probability distributions and confidence diagnostics will appear
                here once a scan is analyzed.
              </p>
            ) : (
              <div className="flex flex-col h-full">
                {result.is_ood && (
                  <div className="bg-orange-50 border-l-4 border-orange-500 p-4 rounded-r-xl mb-5">
                    <strong className="block text-xs font-bold text-orange-900 mb-1">
                      Out-of-Distribution Input Detected
                    </strong>
                    <p className="text-[11px] text-orange-800 leading-relaxed">
                      Model confidence ({result.confidence}%) or prediction
                      entropy ({result.entropy}) indicates this image may not be
                      a valid cranial MRI scan.
                    </p>
                  </div>
                )}

                <div
                  className={`pb-5 mb-5 border-b border-slate-200 ${result.is_ood ? "text-orange-900" : ""}`}
                >
                  <span className="block text-xs font-medium text-slate-400 mb-1">
                    Primary Prediction
                  </span>
                  <div
                    className={`text-2xl font-extrabold tracking-tight ${result.is_ood ? "text-orange-800 text-xl" : "text-[#0c1e35]"}`}
                  >
                    {result.prediction}
                  </div>
                  <div className="flex gap-4 text-xs font-medium text-slate-500 mt-2.5">
                    <span>Confidence: {result.confidence}%</span>
                    <span>Entropy: {result.entropy}</span>
                  </div>
                </div>

                <div className="mb-5">
                  <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-3.5">
                    Class Distribution
                  </h4>
                  <div className="flex flex-col gap-3.5">
                    {result.probabilities.map((item) => (
                      <div key={item.raw_label} className="flex flex-col">
                        <div className="flex justify-between text-xs mb-1.5">
                          <span className="font-medium text-slate-700">
                            {item.label}
                          </span>
                          <span className="font-bold text-slate-900">
                            {item.probability}%
                          </span>
                        </div>
                        <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-gradient-to-r from-[#0c1e35] to-sky-500 rounded-full transition-all duration-500"
                            style={{ width: `${item.probability}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="mt-auto pt-5 flex items-center gap-3 text-[11px] text-slate-400">
                  <span>Low activation</span>
                  <div className="flex-1 h-2 rounded-full bg-gradient-to-r from-blue-900 via-blue-500 via-cyan-400 via-yellow-300 via-red-500 to-rose-900" />
                  <span>High activation</span>
                </div>
              </div>
            )}
          </div>
        </div>
      </section>

      {/* Inspect Mode Modal */}
      {inspectMedia && (
        <div
          className="fixed inset-0 bg-[#070d17]/90 backdrop-blur-md flex items-center justify-center z-50 p-6"
          onClick={() => setInspectMedia(null)}
        >
          <div
            className="bg-[#0c1e35] border border-white/10 rounded-2xl w-[1000px] max-w-[94vw] max-h-[92vh] flex flex-col overflow-hidden shadow-2xl animate-in fade-in zoom-in-95 duration-200"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex justify-between items-center px-8 py-5 border-b border-white/10 gap-4 flex-wrap">
              <div>
                <h3 className="text-white text-base font-bold">
                  {inspectMedia.title}
                </h3>
                {inspectMedia.detail && (
                  <p className="text-slate-400 text-xs mt-1">
                    {inspectMedia.detail}
                  </p>
                )}
              </div>
              <div className="flex items-center gap-3">
                <button
                  onClick={() =>
                    setInspectZoom((z) => Math.max(1, +(z - 0.25).toFixed(2)))
                  }
                  className="bg-white/10 hover:bg-sky-500 text-white w-9 h-9 rounded-xl flex items-center justify-center transition-colors font-bold"
                >
                  −
                </button>
                <span className="text-slate-300 text-xs font-semibold min-w-[48px] text-center">
                  {Math.round(inspectZoom * 100)}%
                </span>
                <button
                  onClick={() =>
                    setInspectZoom((z) => Math.min(2.5, +(z + 0.25).toFixed(2)))
                  }
                  className="bg-white/10 hover:bg-sky-500 text-white w-9 h-9 rounded-xl flex items-center justify-center transition-colors font-bold"
                >
                  +
                </button>
                <button
                  onClick={() => setInspectMedia(null)}
                  className="bg-orange-600 hover:bg-orange-700 text-white px-4 py-2 rounded-xl text-xs font-bold transition-colors ml-3"
                >
                  Exit Inspect Mode
                </button>
              </div>
            </div>
            <div className="bg-[#050911] flex-1 min-h-[460px] max-h-[75vh] overflow-auto flex items-center justify-center p-8">
              <div
                className="relative inline-flex transition-transform origin-center"
                style={{ transform: `scale(${inspectZoom})` }}
              >
                <img
                  src={inspectMedia.baseSrc}
                  alt={inspectMedia.title}
                  className="max-w-[82vw] max-h-[68vh] w-auto h-auto object-contain block rounded-lg"
                />
                {inspectMedia.overlaySrc && (
                  <img
                    src={inspectMedia.overlaySrc}
                    alt="Heatmap overlay"
                    className="absolute inset-0 w-full h-full object-contain mix-blend-screen rounded-lg"
                    style={{ opacity: inspectMedia.overlayOpacity ?? 0.55 }}
                  />
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Footer */}
      <footer className="max-w-7xl mx-auto px-8 py-12 border-t border-slate-200">
        <p className="text-xs text-slate-500 max-w-4xl leading-relaxed">
          Medical Disclaimer: This tool was built for learning purposes and not
          for medical diagnosis.
        </p>
      </footer>
    </div>
  );
}
