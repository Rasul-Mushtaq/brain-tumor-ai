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

export default function App() {
  const [selectedSampleId, setSelectedSampleId] = useState(null);
  const [fileName, setFileName] = useState("");
  const [result, setResult] = useState(null);
  const [activeStage, setActiveStage] = useState("layer4_heatmap");
  const [opacity, setOpacity] = useState(0.55);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [hoveredMedia, setHoveredMedia] = useState(null);
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
      const response = await fetch("http://localhost:8000/analyze", {
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
          ? "Unable to connect to the backend server. Please verify FastAPI is running on http://localhost:8000."
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
    setHoveredMedia(null);
    setInspectZoom(1);
    setInspectMedia(mediaObj);
  };

  const currentStage =
    PIPELINE_STAGES.find((stage) => stage.id === activeStage) ||
    PIPELINE_STAGES[3];

  return (
    <div className="min-h-screen bg-[#f5f7fa] text-[#0c1e35] font-sans antialiased selection:bg-sky-500 selection:text-white">
      {/* Background Blur Veil */}
      <div
        className={`fixed inset-0 bg-[#0c1e35]/40 backdrop-blur-md transition-opacity duration-300 z-40 pointer-events-none ${hoveredMedia ? "opacity-100 pointer-events-auto" : "opacity-0"}`}
      />

      {/* Hover Center Spotlight */}
      {hoveredMedia && !inspectMedia && (
        <div className="fixed inset-0 flex items-center justify-center z-50 pointer-events-none p-8">
          <div className="relative bg-[#070d17] border-2 border-sky-500 rounded-xl p-4 shadow-2xl max-w-4xl max-h-[85vh] flex flex-col items-center animate-in fade-in zoom-in-95 duration-200">
            <img
              src={hoveredMedia.baseSrc}
              alt={hoveredMedia.title}
              className="max-w-[70vw] max-h-[65vh] object-contain rounded"
            />
            {hoveredMedia.overlaySrc && (
              <img
                src={hoveredMedia.overlaySrc}
                alt="Activation overlay"
                className="absolute top-4 left-4 right-4 bottom-14 max-w-[70vw] max-h-[65vh] m-auto object-contain mix-blend-screen rounded pointer-events-none"
                style={{ opacity: hoveredMedia.overlayOpacity ?? 0.55 }}
              />
            )}
            <div className="w-full flex justify-between items-center pt-3 text-slate-100 text-sm">
              <span className="font-semibold">{hoveredMedia.title}</span>
              <span className="text-sky-400 text-xs">
                Click image to enter Inspect Mode
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Navigation */}
      <nav className="max-w-7xl mx-auto px-8 py-7 flex justify-between items-center border-b border-slate-200">
        <a
          href="#top"
          className="text-xl font-bold tracking-tight text-[#0c1e35] flex items-center gap-2"
        >
          <span className="w-2.5 h-2.5 rounded bg-gradient-to-br from-sky-500 to-orange-600 inline-block"></span>
          Classifying Brain Tumors using Computer Vision
        </a>
        <div className="flex gap-8 text-sm font-medium text-slate-600">
          <a
            href="#methodology"
            className="hover:text-sky-600 transition-colors"
          >
            Methodology
          </a>
          <a href="#workbench" className="hover:text-sky-600 transition-colors">
            Diagnostic Workbench
          </a>
        </div>
      </nav>

      {/* Hero Section */}
      <header
        id="top"
        className="max-w-7xl mx-auto px-8 py-18 border-b border-slate-200"
      >
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-14 items-center">
          <div>
            <h1 className="text-4xl lg:text-5xl font-bold tracking-tight text-[#0c1e35] leading-[1.12] mb-6">
              Interpretable cranial MRI classification with multi-layer
              Grad-CAM++.
            </h1>
            <p className="text-lg text-slate-600 leading-relaxed max-w-xl mb-9">
              A fine-tuned ResNet18 vision pipeline that classifies glioma,
              meningioma, pituitary tumors, and healthy scans while exposing
              spatial feature activations across network depths.
            </p>
            <div className="flex gap-4 flex-wrap">
              <button
                onClick={scrollToWorkbench}
                className="bg-sky-600 hover:bg-sky-700 text-white font-semibold px-6 py-3.5 rounded-lg shadow-sm transition-all hover:-translate-y-0.5"
              >
                Launch Interactive Workbench
              </button>
              <a
                href="#methodology"
                className="bg-white hover:bg-sky-50 text-slate-900 border border-slate-300 hover:border-sky-500 font-medium px-6 py-3.5 rounded-lg transition-all"
              >
                Review Architecture
              </a>
            </div>
          </div>

          {/* Asymmetric Showcase */}
          <div className="bg-white border border-slate-200 border-t-4 border-t-sky-500 rounded-xl p-6 shadow-xl">
            <div className="flex flex-col gap-4">
              <figure
                className="bg-[#0a111e] border border-slate-200 rounded-lg p-3 cursor-zoom-in transition-transform hover:-translate-y-1 w-[82%] self-start"
                onMouseEnter={() =>
                  setHoveredMedia({
                    baseSrc: "/brain_tumor_sample.jpg",
                    title: "Raw Cranial MRI Input",
                  })
                }
                onMouseLeave={() => setHoveredMedia(null)}
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
                  className="w-full max-h-[200px] object-contain block rounded"
                />
                <figcaption className="w-full mt-2 text-xs text-slate-300 flex justify-between items-center">
                  <span>Input MRI Scan</span>
                  <span className="text-sky-400 text-[11px]">
                    Hover to enlarge
                  </span>
                </figcaption>
              </figure>

              <figure
                className="bg-[#0a111e] border border-sky-900/30 rounded-lg p-3 cursor-zoom-in transition-transform hover:-translate-y-1 w-[88%] self-end"
                onMouseEnter={() =>
                  setHoveredMedia({
                    baseSrc: "/brain_tumor_heatmap.jpg",
                    title: "Grad-CAM++ Spatial Localization",
                  })
                }
                onMouseLeave={() => setHoveredMedia(null)}
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
                  className="w-full max-h-[200px] object-contain block rounded"
                />
                <figcaption className="w-full mt-2 text-xs text-slate-300 flex justify-between items-center">
                  <span>Grad-CAM++ Localization</span>
                  <span className="text-sky-400 text-[11px]">
                    Hover to enlarge
                  </span>
                </figcaption>
              </figure>
            </div>
            <p className="mt-4 pt-3 border-t border-slate-100 text-xs text-slate-500 text-center">
              Hover any scan to spotlight proportions, or click to lock Inspect
              Mode.
            </p>
          </div>
        </div>
      </header>

      {/* Methodology Section */}
      <section
        id="methodology"
        className="max-w-7xl mx-auto px-8 py-18 border-b border-slate-200"
      >
        <div className="max-w-2xl mb-12">
          <h2 className="text-2xl font-bold tracking-tight text-[#0c1e35] mb-3">
            Pipeline Architecture
          </h2>
          <p className="text-slate-600 text-base">
            Medical imaging requires both spatial precision and calibrated
            uncertainty. Each scan passes through three distinct stages before a
            classification is surfaced.
          </p>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-9">
          <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm">
            <div className="w-9 h-1 bg-sky-500 rounded mb-4"></div>
            <h3 className="text-lg font-semibold text-[#0c1e35] mb-2.5">
              Two-Stage Fine-Tuning
            </h3>
            <p className="text-slate-600 text-sm leading-relaxed">
              The classification head is trained first to stabilize gradients
              before unfreezing residual blocks layer3 and layer4 under a cosine
              annealing schedule, adapting pretrained filters to cranial tissue
              contrast.
            </p>
          </div>
          <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm">
            <div className="w-9 h-1 bg-orange-600 rounded mb-4"></div>
            <h3 className="text-lg font-semibold text-[#0c1e35] mb-2.5">
              Second-Order Localization
            </h3>
            <p className="text-slate-600 text-sm leading-relaxed">
              Grad-CAM++ computes pixel-wise gradient weights across multiple
              convolutional layers, producing sharper boundaries on irregular
              lesions than standard global average pooling.
            </p>
          </div>
          <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm">
            <div className="w-9 h-1 bg-[#0c1e35] rounded mb-4"></div>
            <h3 className="text-lg font-semibold text-[#0c1e35] mb-2.5">
              Out-of-Distribution Guardrail
            </h3>
            <p className="text-slate-600 text-sm leading-relaxed">
              Predictions falling below 65% confidence or exceeding 0.72
              normalized prediction entropy are flagged as uncertain, preventing
              forced classifications on non-MRI uploads.
            </p>
          </div>
        </div>
      </section>

      {/* Diagnostic Workbench Section */}
      <section
        id="workbench"
        ref={workbenchRef}
        className="max-w-7xl mx-auto px-8 py-18"
      >
        <div className="flex justify-between items-end gap-6 mb-8 flex-wrap">
          <div>
            <h2 className="text-2xl font-bold tracking-tight text-[#0c1e35] mb-2">
              Diagnostic Workbench
            </h2>
            <p className="text-slate-600 text-sm">
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
              className="bg-[#0c1e35] hover:bg-[#163254] text-white font-medium px-5 py-2.5 rounded-lg text-sm transition-colors disabled:opacity-50"
            >
              Upload Custom Scan
            </button>
          </div>
        </div>

        {/* Sample Gallery Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5 mb-8">
          {SAMPLE_SCANS.map((sample) => (
            <button
              key={sample.id}
              onClick={() => handleSampleSelect(sample)}
              disabled={loading}
              className={`flex flex-col bg-white border rounded-xl p-4 text-left transition-all shadow-sm hover:-translate-y-0.5 hover:shadow-md ${selectedSampleId === sample.id ? "border-sky-500 border-2 bg-sky-50/50" : "border-slate-200"}`}
            >
              <div className="w-full h-44 bg-[#070d17] rounded-lg overflow-hidden flex items-center justify-center mb-3">
                <img
                  src={sample.src}
                  alt={sample.title}
                  className="w-full h-full object-contain"
                />
              </div>
              <strong className="text-sm font-semibold text-[#0c1e35]">
                {sample.title}
              </strong>
              <span className="text-xs text-slate-500 mt-0.5">
                {sample.note}
              </span>
            </button>
          ))}
        </div>

        {error && (
          <div className="bg-red-50 border border-red-500 text-red-700 px-4 py-3 rounded-lg text-sm mb-6">
            {error}
          </div>
        )}

        {/* Three Column Inspector */}
        <div className="grid grid-cols-1 lg:grid-cols-[290px_1fr_310px] gap-6 items-stretch">
          {/* Stage Selector */}
          <div className="bg-white border border-slate-200 rounded-xl p-6 flex flex-col shadow-sm">
            <h3 className="text-sm font-bold text-[#0c1e35] mb-4">
              Activation Depth
            </h3>
            <div className="flex flex-col gap-2.5">
              {PIPELINE_STAGES.map((stage) => (
                <button
                  key={stage.id}
                  onClick={() => setActiveStage(stage.id)}
                  disabled={!result || loading}
                  className={`flex items-start gap-3 p-3.5 bg-slate-50 border rounded-lg text-left transition-all ${activeStage === stage.id ? "border-sky-500 bg-sky-50/50" : "border-slate-200 hover:border-slate-300"} disabled:opacity-50`}
                >
                  <span
                    className={`text-xs font-bold pt-0.5 ${activeStage === stage.id ? "text-sky-600" : "text-slate-400"}`}
                  >
                    {stage.index}
                  </span>
                  <div className="flex flex-col">
                    <strong className="text-sm font-semibold text-[#0c1e35]">
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
          <div className="bg-white border border-slate-200 rounded-xl p-6 flex flex-col shadow-sm">
            <div className="flex justify-between items-center mb-4 gap-3 min-h-[28px]">
              <span className="text-xs font-medium text-slate-600">
                {loading
                  ? "Processing scan through inference engine..."
                  : fileName
                    ? `Active scan: ${fileName}`
                    : "No scan loaded"}
              </span>
              {result && activeStage !== "original" && (
                <div className="flex items-center gap-2.5 text-xs text-slate-600">
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

            <div className="bg-[#070d17] rounded-lg min-h-[380px] flex items-center justify-center p-6 mb-5">
              {!result ? (
                <div className="text-slate-400 text-sm text-center max-w-[280px]">
                  Choose any sample scan above or upload an image to begin
                  inspection.
                </div>
              ) : (
                <div
                  className="relative inline-flex items-center justify-center max-w-full max-h-[340px] rounded-md overflow-hidden cursor-zoom-in border border-white/10 transition-transform hover:scale-[1.02]"
                  onMouseEnter={() =>
                    setHoveredMedia({
                      baseSrc: result.images.original,
                      overlaySrc:
                        activeStage !== "original"
                          ? result.images[activeStage]
                          : null,
                      overlayOpacity: opacity,
                      title: `${currentStage.name} — ${fileName}`,
                    })
                  }
                  onMouseLeave={() => setHoveredMedia(null)}
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
                    className="max-w-full max-h-[340px] w-auto h-auto object-contain block"
                  />
                  {activeStage !== "original" && (
                    <img
                      src={result.images[activeStage]}
                      alt="Grad-CAM++ Activation"
                      className="absolute inset-0 w-full h-full object-contain mix-blend-screen transition-opacity"
                      style={{ opacity }}
                    />
                  )}
                </div>
              )}
            </div>

            <div className="border-t border-slate-200 pt-4 mt-auto">
              <strong className="block text-sm font-semibold text-[#0c1e35] mb-1">
                {currentStage.name}
              </strong>
              <p className="text-xs text-slate-600 leading-relaxed">
                {currentStage.summary}
              </p>
            </div>
          </div>

          {/* Diagnostic Output */}
          <div className="bg-white border border-slate-200 rounded-xl p-6 flex flex-col shadow-sm">
            <h3 className="text-sm font-bold text-[#0c1e35] mb-4">
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
                  <div className="bg-orange-50 border-l-4 border-orange-500 p-3 rounded-r-lg mb-4">
                    <strong className="block text-xs font-semibold text-orange-900 mb-1">
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
                  className={`pb-4 mb-4 border-b border-slate-200 ${result.is_ood ? "text-orange-900" : ""}`}
                >
                  <span className="block text-xs text-slate-400 mb-1">
                    Primary Prediction
                  </span>
                  <div
                    className={`text-xl font-bold tracking-tight ${result.is_ood ? "text-orange-800 text-lg" : "text-[#0c1e35]"}`}
                  >
                    {result.prediction}
                  </div>
                  <div className="flex gap-4 text-xs text-slate-500 mt-2">
                    <span>Confidence: {result.confidence}%</span>
                    <span>Entropy: {result.entropy}</span>
                  </div>
                </div>

                <div className="mb-4">
                  <h4 className="text-xs font-semibold text-slate-600 mb-3">
                    Class Distribution
                  </h4>
                  <div className="flex flex-col gap-3">
                    {result.probabilities.map((item) => (
                      <div key={item.raw_label} className="flex flex-col">
                        <div className="flex justify-between text-xs mb-1">
                          <span className="text-slate-700">{item.label}</span>
                          <span className="font-medium text-slate-900">
                            {item.probability}%
                          </span>
                        </div>
                        <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-gradient-to-r from-[#0c1e35] to-sky-500 rounded-full transition-all duration-300"
                            style={{ width: `${item.probability}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="mt-auto pt-4 flex items-center gap-2 text-[11px] text-slate-400">
                  <span>Low activation</span>
                  <div className="flex-1 h-1.5 rounded-full bg-gradient-to-r from-blue-900 via-blue-500 via-cyan-400 via-yellow-300 via-red-500 to-rose-900" />
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
            className="bg-[#0c1e35] border border-white/10 rounded-xl w-[960px] max-w-[94vw] max-h-[92vh] flex flex-col overflow-hidden shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex justify-between items-center px-6 py-4 border-b border-white/10 gap-4 flex-wrap">
              <div>
                <h3 className="text-white text-base font-semibold">
                  {inspectMedia.title}
                </h3>
                {inspectMedia.detail && (
                  <p className="text-slate-400 text-xs mt-0.5">
                    {inspectMedia.detail}
                  </p>
                )}
              </div>
              <div className="flex items-center gap-2.5">
                <button
                  onClick={() =>
                    setInspectZoom((z) => Math.max(1, +(z - 0.25).toFixed(2)))
                  }
                  className="bg-white/10 hover:bg-sky-500 text-white w-8 h-8 rounded-lg flex items-center justify-center transition-colors"
                >
                  −
                </button>
                <span className="text-slate-300 text-xs min-w-[44px] text-center">
                  {Math.round(inspectZoom * 100)}%
                </span>
                <button
                  onClick={() =>
                    setInspectZoom((z) => Math.min(2.5, +(z + 0.25).toFixed(2)))
                  }
                  className="bg-white/10 hover:bg-sky-500 text-white w-8 h-8 rounded-lg flex items-center justify-center transition-colors"
                >
                  +
                </button>
                <button
                  onClick={() => setInspectMedia(null)}
                  className="bg-orange-600 hover:bg-orange-700 text-white px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors ml-2"
                >
                  Exit Inspect Mode
                </button>
              </div>
            </div>
            <div className="bg-[#050911] flex-1 min-h-[440px] max-h-[75vh] overflow-auto flex items-center justify-center p-6">
              <div
                className="relative inline-flex transition-transform origin-center"
                style={{ transform: `scale(${inspectZoom})` }}
              >
                <img
                  src={inspectMedia.baseSrc}
                  alt={inspectMedia.title}
                  className="max-w-[82vw] max-h-[68vh] w-auto h-auto object-contain block rounded"
                />
                {inspectMedia.overlaySrc && (
                  <img
                    src={inspectMedia.overlaySrc}
                    alt="Heatmap overlay"
                    className="absolute inset-0 w-full h-full object-contain mix-blend-screen rounded"
                    style={{ opacity: inspectMedia.overlayOpacity ?? 0.55 }}
                  />
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Footer */}
      <footer className="max-w-7xl mx-auto px-8 py-10 border-t border-slate-200">
        <p className="text-xs text-slate-500 max-w-4xl leading-relaxed">
          Medical Disclaimer: This project and tool was built for learning
          purposes and not for diagnosis.
        </p>
      </footer>
    </div>
  );
}
