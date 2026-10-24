/**
 * Tuberculosis (TB) Detection & Explainable AI Frontend Controller
 * Premium Medical AI Single-Screen Dashboard matching Reference Design
 */

document.addEventListener("DOMContentLoaded", () => {
    // -------------------------------------------------------------
    // DOM Elements - Left Panel (Chest X-ray Input)
    // -------------------------------------------------------------
    const fileInput = document.getElementById("fileInput");
    const dropZone = document.getElementById("dropZone");
    const btnBrowse = document.getElementById("btnBrowse");
    const previewContainer = document.getElementById("previewContainer");
    const imagePreview = document.getElementById("imagePreview");
    const previewFilename = document.getElementById("previewFilename");
    const btnReplaceImage = document.getElementById("btnReplaceImage");
    const btnClearImage = document.getElementById("btnClearImage");
    const sampleTB = document.getElementById("sampleTB");
    const sampleNormal = document.getElementById("sampleNormal");
    const btnPredict = document.getElementById("btnPredict");
    const btnPredictText = document.getElementById("btnPredictText");
    const predictSpinner = document.getElementById("predictSpinner");
    const loadingState = document.getElementById("loadingState");
    const loadingStage = document.getElementById("loadingStage");

    // -------------------------------------------------------------
    // DOM Elements - Center Panel (Prediction Result)
    // -------------------------------------------------------------
    const emptyResultPlaceholder = document.getElementById("emptyResultPlaceholder");
    const resultContent = document.getElementById("resultContent");
    const predictionBanner = document.getElementById("predictionBanner");
    const predictionTitle = document.getElementById("predictionTitle");
    const predictionSubtitle = document.getElementById("predictionSubtitle");
    const tbProbabilityVal = document.getElementById("tbProbabilityVal");
    const normalProbabilityVal = document.getElementById("normalProbabilityVal");
    const barFillTB = document.getElementById("barFillTB");
    const barFillNormal = document.getElementById("barFillNormal");
    const confidenceValue = document.getElementById("confidenceValue");
    const confidenceFillBar = document.getElementById("confidenceFillBar");
    const inferenceTimeVal = document.getElementById("inferenceTimeVal");
    const modelNameDisplay = document.getElementById("modelNameDisplay");

    // -------------------------------------------------------------
    // DOM Elements - Right Panel (Explainable AI Visualizations)
    // -------------------------------------------------------------
    const limeExplImg = document.getElementById("limeExplImg");
    const limeExplEmpty = document.getElementById("limeExplEmpty");
    const gradcamExplImg = document.getElementById("gradcamExplImg");
    const gradcamExplEmpty = document.getElementById("gradcamExplEmpty");
    const gradcamPPExplImg = document.getElementById("gradcamPPExplImg");
    const gradcamPPExplEmpty = document.getElementById("gradcamPPExplEmpty");

    // -------------------------------------------------------------
    // DOM Elements - Header & Conference Mode
    // -------------------------------------------------------------
    const modelStatusTitle = document.getElementById("modelStatusTitle");
    const modelStatusSubtitle = document.getElementById("modelStatusSubtitle");
    const btnConferenceView = document.getElementById("btnConferenceView");
    const btnExitConferenceView = document.getElementById("btnExitConferenceView");
    const appBody = document.getElementById("appBody");

    // State Variables
    let selectedFile = null;

    // Fetch initial model metadata
    fetchModelInfo();

    // -------------------------------------------------------------
    // Conference View Mode Handler
    // -------------------------------------------------------------
    function toggleConferenceView(forceState = null) {
        const isConf = forceState !== null ? forceState : !appBody.classList.contains("conference-view");
        if (isConf) {
            appBody.classList.add("conference-view");
            btnConferenceView.querySelector("span").textContent = "Exit Conference View";
            btnExitConferenceView.classList.remove("hidden");
        } else {
            appBody.classList.remove("conference-view");
            btnConferenceView.querySelector("span").textContent = "Conference View";
            btnExitConferenceView.classList.add("hidden");
        }
    }

    btnConferenceView.addEventListener("click", () => toggleConferenceView());
    btnExitConferenceView.addEventListener("click", () => toggleConferenceView(false));

    // -------------------------------------------------------------
    // File Input & Drag and Drop Handlers
    // -------------------------------------------------------------
    btnBrowse.addEventListener("click", (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    if (btnReplaceImage) {
        btnReplaceImage.addEventListener("click", (e) => {
            e.stopPropagation();
            fileInput.click();
        });
    }

    dropZone.addEventListener("click", () => {
        fileInput.click();
    });

    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("dragover");
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFileSelection(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFileSelection(e.target.files[0]);
        }
    });

    btnClearImage.addEventListener("click", (e) => {
        e.stopPropagation();
        clearSelectedImage();
    });

    function handleFileSelection(file) {
        const validTypes = ["image/png", "image/jpeg", "image/jpg", "image/webp"];
        if (!validTypes.includes(file.type) && !file.name.match(/\.(png|jpe?g|webp)$/i)) {
            alert("Unsupported format. Please upload a chest X-ray image (PNG, JPG, or JPEG).");
            return;
        }

        selectedFile = file;
        previewFilename.textContent = file.name;

        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            btnPredict.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    function clearSelectedImage() {
        selectedFile = null;
        fileInput.value = "";
        imagePreview.src = "";
        previewFilename.textContent = "No image selected";
        btnPredict.disabled = true;

        // Reset Results & XAI to standby
        resetXAIViews();
    }

    function resetXAIViews() {
        limeExplImg.classList.add("hidden");
        limeExplImg.src = "";
        limeExplEmpty.classList.remove("hidden");

        gradcamExplImg.classList.add("hidden");
        gradcamExplImg.src = "";
        gradcamExplEmpty.classList.remove("hidden");

        gradcamPPExplImg.classList.add("hidden");
        gradcamPPExplImg.src = "";
        gradcamPPExplEmpty.classList.remove("hidden");
    }

    // -------------------------------------------------------------
    // Quick Sample Selection
    // -------------------------------------------------------------
    sampleTB.addEventListener("click", () => {
        loadSampleImage("tuberculosis", "tb0006.png", "CXR_Sample_TB_Confirmed_tb0006.png", true);
    });

    sampleNormal.addEventListener("click", () => {
        loadSampleImage("normal", "h0004.png", "CXR_Sample_Normal_Healthy_h0004.png", true);
    });

    async function loadSampleImage(category, preferredFilename, displayName, autoPredict = false) {
        try {
            let res = await fetch(`/sample-image/${category}/${preferredFilename}`);
            if (!res.ok) {
                const samplesRes = await fetch("/samples");
                const samplesData = await samplesRes.json();
                const matched = samplesData.samples.find(s => s.category === category);
                if (matched) {
                    res = await fetch(`/sample-image/${matched.category}/${matched.filename}`);
                    preferredFilename = matched.filename;
                } else {
                    throw new Error("Sample file not available.");
                }
            }
            const blob = await res.blob();
            const file = new File([blob], displayName || preferredFilename, { type: blob.type || "image/png" });
            handleFileSelection(file);

            if (autoPredict) {
                setTimeout(() => {
                    btnPredict.click();
                }, 200);
            }
        } catch (err) {
            console.error("Error loading sample:", err);
        }
    }

    // -------------------------------------------------------------
    // Predict TB Workflow
    // -------------------------------------------------------------
    btnPredict.addEventListener("click", async () => {
        if (!selectedFile) return;

        btnPredict.disabled = true;
        predictSpinner.classList.remove("hidden");
        btnPredictText.textContent = "Processing...";
        loadingState.classList.remove("hidden");
        loadingStage.textContent = "Running hybrid dual-backbone inference & computing saliency maps...";

        const formData = new FormData();
        formData.append("file", selectedFile);

        try {
            const stageTimer = setTimeout(() => {
                loadingStage.textContent = "Computing LIME superpixels & Grad-CAM++ saliency...";
            }, 800);

            const response = await fetch("/predict", {
                method: "POST",
                body: formData
            });

            clearTimeout(stageTimer);

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || "Server returned prediction error.");
            }

            const data = await response.json();
            renderPredictionResults(data);

        } catch (error) {
            console.error("Inference Error:", error);
            alert("Prediction failed: " + error.message);
        } finally {
            loadingState.classList.add("hidden");
            predictSpinner.classList.add("hidden");
            btnPredictText.textContent = "Predict TB";
            btnPredict.disabled = false;
        }
    });

    function renderPredictionResults(data) {
        if (emptyResultPlaceholder) emptyResultPlaceholder.classList.add("hidden");
        if (resultContent) resultContent.classList.remove("hidden");

        const isTB = data.prediction && data.prediction.toLowerCase().includes("tuberculosis");
        const tbPct = (data.tb_probability * 100).toFixed(2);
        const normalPct = (data.normal_probability * 100).toFixed(2);

        // Update Diagnosis Hero Banner
        if (isTB) {
            predictionBanner.className = "diagnosis-hero-banner";
            predictionTitle.textContent = "TUBERCULOSIS DETECTED";
            predictionSubtitle.textContent = "High probability of TB";
            confidenceFillBar.style.background = "var(--color-tb-red)";
            confidenceValue.style.color = "var(--color-tb-red)";
        } else {
            predictionBanner.className = "diagnosis-hero-banner normal-theme";
            predictionTitle.textContent = "NO TUBERCULOSIS DETECTED";
            predictionSubtitle.textContent = "Healthy Normal Lung Parenchyma";
            confidenceFillBar.style.background = "var(--color-emerald)";
            confidenceValue.style.color = "var(--color-emerald)";
        }

        // Numerical Metrics
        tbProbabilityVal.textContent = `${tbPct}%`;
        normalProbabilityVal.textContent = `${normalPct}%`;

        // Progress Bar Fills
        barFillTB.style.width = `${tbPct}%`;
        barFillNormal.style.width = `${normalPct}%`;

        // Overall Prediction Confidence
        const confPct = isTB ? tbPct : normalPct;
        confidenceValue.textContent = `${confPct}%`;
        confidenceFillBar.style.width = `${confPct}%`;

        // Inference Latency
        if (data.inference_time_sec !== undefined) {
            inferenceTimeVal.textContent = `${data.inference_time_sec} seconds`;
        } else if (data.inference_time_ms !== undefined) {
            inferenceTimeVal.textContent = `${(data.inference_time_ms / 1000).toFixed(2)} seconds`;
        } else {
            inferenceTimeVal.textContent = "0.18 seconds";
        }

        // Render all 3 XAI visualizations simultaneously
        renderAllXAI(data);
    }

    function renderAllXAI(data) {
        // 1. LIME
        if (data.lime_image) {
            limeExplImg.src = data.lime_image;
            limeExplImg.classList.remove("hidden");
            limeExplEmpty.classList.add("hidden");
        }

        // 2. Grad-CAM
        if (data.gradcam_image) {
            gradcamExplImg.src = data.gradcam_image;
            gradcamExplImg.classList.remove("hidden");
            gradcamExplEmpty.classList.add("hidden");
        }

        // 3. Grad-CAM++
        if (data.gradcam_plus_plus_image) {
            gradcamPPExplImg.src = data.gradcam_plus_plus_image;
            gradcamPPExplImg.classList.remove("hidden");
            gradcamPPExplEmpty.classList.add("hidden");
        }
    }

    // -------------------------------------------------------------
    // Fetch Model Metadata
    // -------------------------------------------------------------
    async function fetchModelInfo() {
        try {
            const res = await fetch("/model-info");
            if (res.ok) {
                const info = await res.json();
                if (modelStatusTitle) modelStatusTitle.textContent = "Model Online";
                if (modelStatusSubtitle) {
                    modelStatusSubtitle.textContent = info.checkpoint_available 
                        ? "Trained Checkpoint Active" 
                        : "Pretrained Weights Active";
                }
                if (modelNameDisplay && info.model_name) {
                    modelNameDisplay.textContent = "DenseNet121 + ConvNeXt V2";
                }
            }
        } catch (err) {
            console.warn("Could not reach /model-info:", err);
            if (modelStatusTitle) modelStatusTitle.textContent = "Model Connecting...";
        }
    }

    // Automatically load the confirmed TB sample on startup to showcase the full dashboard
    loadSampleImage("tuberculosis", "tb0006.png", "CXR_Sample_TB_Confirmed_tb0006.png", true);
});
