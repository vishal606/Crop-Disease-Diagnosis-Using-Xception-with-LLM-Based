# Explainable Crop Disease Diagnosis Using Xception with LLM-Based Natural Language Decision Support for Bangladeshi Crops

An end-to-end, explainable deep learning pipeline for diagnosing crop diseases in Bangladeshi agriculture. A
fine-tuned **Xception** CNN classifies leaf images, **Grad-CAM** and **SHAP** explain *why* the model made its
prediction, and an **LLM-based module** turns that technical output into a plain-language diagnostic report with
treatment and prevention advice. The full pipeline is also wrapped in an interactive **Streamlit** app.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Contents](#repository-contents)
- [Results Summary](#results-summary)
- [Dataset](#dataset)
- [Pipeline Architecture](#pipeline-architecture)
- [Getting Started](#getting-started)
  - [Requirements](#requirements)
  - [Running the Notebook](#running-the-notebook)
  - [Running the Streamlit App](#running-the-streamlit-app)
  - [Recompiling the Thesis PDF](#recompiling-the-thesis-pdf)
- [LLM Backend Configuration](#llm-backend-configuration)
- [Known Limitations](#known-limitations)
- [Future Work](#future-work)
- [Citation](#citation)

---

## Project Overview

| Component      | Method                                      |
|----------------|----------------------------------------------|
| Dataset        | New Bangladeshi Crop Disease (Kaggle)         |
| Classifier     | Xception (transfer learning + fine-tuning)    |
| Explainability | Grad-CAM + SHAP                               |
| LLM            | OpenAI GPT, local Llama-3/Gemma, or template fallback |
| Backend        | Python, TensorFlow / Keras                    |
| Interface      | Streamlit                                     |

The system takes a photo of a crop leaf and returns:
1. The predicted disease (or healthy) class and confidence score
2. A Grad-CAM heatmap showing which regions of the leaf drove the prediction
3. A SHAP attribution map as a second, independent explanation
4. A natural-language report covering **disease explanation**, **confidence interpretation**,
   **recommended treatment**, and **preventive measures**

## Repository Contents

```
.
├── Explainable_Crop_Disease_Diagnosis_Xception_LLM.ipynb   # Main thesis notebook (data → model → XAI → LLM)
├── Thesis_Explainable_Crop_Disease_Diagnosis.pdf            # Compiled thesis (LaTeX, 41 pages)
├── Thesis_Explainable_Crop_Disease_Diagnosis.tex            # Thesis LaTeX source
├── Thesis_LaTeX_Source_Bundle.zip                           # .tex source + all figures, for local recompilation
├── app.py                                                   # Standalone Streamlit app (written by the notebook)
├── figures/                                                 # Exported result figures used in the thesis
│   ├── class_distribution.png
│   ├── sample_grid.png
│   ├── split_distribution.png
│   ├── augmentation_examples.png
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   ├── roc_curves.png
│   ├── gradcam_demo.png
│   ├── shap_demo.png
│   ├── explanation_comparison.png
│   └── interactive_demo_result.png
├── models/                                                  # Created after training (not included by default)
│   ├── xception_bd_crop_best.keras
│   └── class_indices.json
└── results/                                                 # Created after running the notebook
    ├── predictions.csv
    ├── explanation_reports.json
    ├── final_metrics.json
    └── heatmaps/
```

> **Note:** `models/` and most of `results/` are generated when you run the notebook end-to-end; they are not
> pre-populated in this repository because trained model weights are large binary files. The `figures/` folder
> does contain the actual PNGs produced by a completed training/evaluation run, which is what the thesis PDF
> embeds.

## Results Summary

Trained and evaluated on the New Bangladeshi Crop Disease dataset (13,024 images, 14 classes across Corn,
Potato, Rice, and Wheat; stratified 70/15/15 split → 9,116 train / 1,954 val / 1,954 test):

| Metric                        | Value  |
|--------------------------------|--------|
| Test Accuracy                  | 92.27% |
| Macro-averaged Precision       | 0.9101 |
| Macro-averaged Recall          | 0.9140 |
| Macro-averaged F1-score        | 0.9108 |
| Weighted-averaged Precision    | 0.9251 |
| Weighted-averaged Recall       | 0.9227 |
| Weighted-averaged F1-score     | 0.9229 |
| Macro-averaged ROC-AUC (14 classes) | 0.9969 |

Full per-class precision/recall/F1, the confusion matrix, ROC curves, and qualitative Grad-CAM/SHAP comparisons
are in Chapter 4 of the thesis PDF and in Section 5–6 of the notebook.

The model performs strongly across nearly all classes (e.g., Rice–Neck Blast F1 = 1.000, Corn–Common Rust
F1 = 0.997), with the main confusion occurring between **Rice–Brown Spot** and **Rice–Leaf Blast**, two visually
similar rice diseases — see the Discussion section of the thesis for analysis and suggested mitigations.

## Dataset

- **Source:** New Bangladeshi Crop Disease dataset (Kaggle)
- **Size:** 13,024 leaf images
- **Classes (14):** Corn (Common Rust, Gray Leaf Spot, Healthy, Northern Leaf Blight), Potato (Early Blight,
  Healthy, Late Blight), Rice (Brown Spot, Healthy, Leaf Blast, Neck Blast), Wheat (Brown Rust, Healthy,
  Yellow Rust)
- **Split:** Stratified 70% train / 15% validation / 15% test

To use the notebook, download the dataset from Kaggle (or attach it directly if running inside a Kaggle
Notebook via "Add Data") — see Section 1.2 of the notebook for the exact loading logic, including automatic
dataset-root detection.

## Pipeline Architecture

```
Leaf Image → Preprocess (299×299, augment, normalize)
           → Xception CNN (transfer learning + fine-tuning)
           → Predicted Class + Confidence
           → Grad-CAM + SHAP explanations
           → Prompt generation (prediction + XAI cues + agronomic knowledge base)
           → LLM (GPT / Llama / Gemma / template fallback)
           → Natural-language report (explanation, confidence, treatment, prevention)
           → Streamlit interactive interface
```

Training uses a **two-phase transfer-learning strategy**:
1. **Phase 1:** Xception backbone frozen; train only the new classification head (Adam, lr = 1e-3).
2. **Phase 2:** Unfreeze the top Xception blocks (from layer index 100 onward); fine-tune end-to-end at a much
   lower learning rate (Adam, lr = 1e-5), with EarlyStopping, ModelCheckpoint, and ReduceLROnPlateau callbacks.

## Getting Started

### Requirements

- Python 3.10+
- GPU strongly recommended for training (Sections 3–4 of the notebook)
- Key packages: `tensorflow`, `shap`, `opencv-python-headless`, `scikit-learn`, `matplotlib`, `seaborn`,
  `pandas`, `numpy`, `pillow`, `streamlit`, `openai` (optional), `kaggle` (optional, for dataset download)

Install everything with:
```bash
pip install tensorflow shap opencv-python-headless scikit-learn matplotlib seaborn \
            pandas numpy pillow streamlit openai kaggle
```

### Running the Notebook

1. Open `Explainable_Crop_Disease_Diagnosis_Xception_LLM.ipynb` in Jupyter, Colab, or Kaggle Notebooks.
2. Download or attach the dataset (Section 1.2).
3. Run all cells sequentially — Sections 1–2 handle data loading/preprocessing, Sections 3–4 train the model,
   Section 5 evaluates it, Section 6 generates Grad-CAM/SHAP explanations, Section 7 wires up the LLM module,
   Section 8 provides an interactive upload widget, and Section 9 saves all results (`predictions.csv`,
   `explanation_reports.json`, heatmap images) to `results/`.
4. The best model is saved to `models/xception_bd_crop_best.keras`, along with `models/class_indices.json`.

### Running the Streamlit App

The notebook's Appendix (Section 11) writes a standalone `app.py`. Once you have a trained model saved under
`models/`, run:
```bash
streamlit run app.py
```
Upload a leaf photo to see the prediction, Grad-CAM heatmap, SHAP map, and generated report in one view.

### Recompiling the Thesis PDF

Unzip `Thesis_LaTeX_Source_Bundle.zip` (contains `main.tex` and all figure PNGs in one folder), then:
```bash
pdflatex main.tex
pdflatex main.tex   # run twice to resolve cross-references, TOC, and citations
```
Before compiling, fill in the placeholder fields near the top of `main.tex` (author name, student ID, supervisor,
university, submission date):
```latex
\newcommand{\authorname}{[Author Name]}
\newcommand{\studentid}{[Student ID]}
\newcommand{\supervisorname}{[Supervisor Name, Designation]}
\newcommand{\universityname}{[University Name]}
\newcommand{\submissiondate}{[Month, Year]}
```

## LLM Backend Configuration

The natural-language decision-support module (Section 7 of the notebook) supports three interchangeable
backends, selected automatically or via the `CROP_LLM_BACKEND` environment variable:

| Backend    | How to enable                                                             |
|------------|-----------------------------------------------------------------------------|
| OpenAI GPT | Set `OPENAI_API_KEY` in your environment (used automatically if present) |
| Local Llama-3 / Gemma | Set `CROP_LLM_BACKEND=local`; requires `transformers` and local model weights (or an `ollama` server) |
| Template (offline) | Default fallback — no setup required, guarantees the pipeline runs fully offline |

The knowledge base grounding the LLM's treatment/prevention advice (`DISEASE_KB` in the notebook and `app.py`) is
illustrative and currently covers a subset of classes in detail; extend it with verified guidance (e.g., from
BARI, BRRI, or the Bangladesh Department of Agricultural Extension) before any real-world deployment.

## Known Limitations

- The dataset, while sizeable, covers only 14 classes across 4 crops; many economically important Bangladeshi
  crops/diseases are not represented.
- Dataset images may not fully reflect real field-capture conditions (lighting, background), so a real-world
  domain shift is possible.
- The agronomic knowledge base is illustrative, not exhaustive or expert-validated for every class.
- Grad-CAM/SHAP agreement is supporting evidence for model reliability, not formal proof of causal biological
  reasoning.
- No formal user study with farmers/extension officers has been conducted to quantify trust/usability gains.

See Chapter 5 of the thesis PDF for the full discussion.

## Future Work

- Expand the dataset with field-captured Bangladeshi crop images to reduce domain shift.
- Validate and extend the agronomic knowledge base with DAE/BARI/BRRI guidance.
- Conduct a human evaluation study with farmers and extension officers.
- Deploy a quantized (TensorFlow Lite) version for offline, on-device inference.
- Address the Rice–Brown Spot / Rice–Leaf Blast confusion via targeted data collection or hierarchical
  classification.
- Explore multimodal vision-language models for richer, image-grounded LLM explanations.

## Citation

If you use this work, please cite it as:

```
[Author Name]. (2026). Explainable Crop Disease Diagnosis Using Xception with LLM-Based Natural Language
Decision Support for Bangladeshi Crops. M.Sc. Thesis, [University Name].
```

---

*This README accompanies the thesis notebook, LaTeX source, and compiled PDF produced for this project. See the
thesis PDF for full methodology, related work, and discussion.*
