"""
Streamlit app: Explainable Crop Disease Diagnosis (Xception + Grad-CAM + SHAP + LLM)

Run with:  streamlit run app.py

This script re-implements the notebook's diagnose_leaf_image() pipeline as a standalone app so it can be run
without a live notebook kernel. It expects:
  - models/xception_bd_crop_best.keras   (trained model from Section 4)
  - models/class_indices.json            (class-name <-> index mapping from Section 4)
"""

import os
import json
import numpy as np
import cv2
import streamlit as st
import matplotlib.cm as cm
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

MODEL_PATH = "models/xception_bd_crop_best.keras"
CLASS_INDICES_PATH = "models/class_indices.json"
IMG_SIZE = (299, 299)

# ---------------- Disease knowledge base (mirrors Section 7.1) ----------------
DISEASE_KB = {
    "Rice_Leaf_Blast": {
        "symptoms": "elongated, spindle-shaped lesions with gray/white centers and brown margins",
        "treatment": "apply a systemic fungicide (e.g. tricyclazole-based); remove heavily infected leaves; "
                     "avoid excess nitrogen fertilizer",
        "prevention": "use blast-resistant varieties; balanced fertilization; good field drainage; field sanitation",
    },
    "Rice_Brown_Spot": {
        "symptoms": "small circular to oval brown lesions with a yellow halo",
        "treatment": "apply appropriate fungicide; correct potassium/nitrogen deficiencies",
        "prevention": "certified disease-free seed; balanced soil fertility; proper water management",
    },
    "Potato_Early_Blight": {
        "symptoms": "dark brown lesions with concentric 'target-board' rings on older leaves",
        "treatment": "protectant fungicide (chlorothalonil/mancozeb-based); remove infected foliage",
        "prevention": "crop rotation; adequate plant spacing for airflow; balanced fertilization",
    },
    "Potato_Late_Blight": {
        "symptoms": "water-soaked, rapidly-expanding dark lesions with white fungal growth underneath",
        "treatment": "systemic fungicide urgently (metalaxyl-based); remove and destroy infected plants",
        "prevention": "resistant varieties; avoid overhead irrigation; monitor closely in cool, wet weather",
    },
}
DEFAULT_KB_ENTRY = {
    "symptoms": "visible discoloration, lesions, or irregular texture consistent with the predicted disease",
    "treatment": "consult local agricultural extension services; remove and destroy infected plant material",
    "prevention": "disease-resistant varieties; crop rotation; balanced fertilization; regular field monitoring",
}


@st.cache_resource
def load_model_and_classes():
    model = keras.models.load_model(MODEL_PATH)
    with open(CLASS_INDICES_PATH) as f:
        class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}
    return model, idx_to_class


def find_last_conv_layer(model):
    backbone = model.get_layer("xception")
    for layer in reversed(backbone.layers):
        if "conv" in layer.name.lower():
            return layer.name
    raise ValueError("No conv layer found.")


def load_and_preprocess(pil_image):
    pil_image = pil_image.convert("RGB").resize(IMG_SIZE)
    raw_arr = img_to_array(pil_image).astype("uint8")
    input_arr = preprocess_input(np.expand_dims(raw_arr.astype("float32"), axis=0))
    return raw_arr, input_arr


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # Keras 3 nested-model fix: build the conv/backbone-output model purely from the backbone's OWN graph,
    # then manually replay the head layers, instead of slicing tensors out of the outer model's graph
    # (which raises KeyError: ...tensor_dict[id(x)] when Xception is nested as a sub-model/layer).
    backbone = model.get_layer("xception")
    conv_layer = backbone.get_layer(last_conv_layer_name)
    conv_and_backbone_output_model = keras.models.Model(
        backbone.input, [conv_layer.output, backbone.output]
    )

    head_layers, seen_backbone = [], False
    for layer in model.layers:
        if layer.name == "xception":
            seen_backbone = True
            continue
        if seen_backbone:
            head_layers.append(layer)

    with tf.GradientTape() as tape:
        conv_outputs, backbone_features = conv_and_backbone_output_model(img_array, training=False)
        tape.watch(conv_outputs)
        x = backbone_features
        for layer in head_layers:
            x = layer(x, training=False)
        predictions = x
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), int(pred_index), predictions.numpy()[0]


def overlay_gradcam(raw_img, heatmap, alpha=0.4):
    heatmap_resized = cv2.resize(heatmap, (raw_img.shape[1], raw_img.shape[0]))
    heatmap_colored = (cm.jet(heatmap_resized)[:, :, :3] * 255).astype("uint8")
    overlay = cv2.addWeighted(raw_img, 1 - alpha, heatmap_colored, alpha, 0)
    return overlay, heatmap_resized


def interpret_confidence(pct):
    if pct >= 90:
        return "very high confidence"
    elif pct >= 75:
        return "high confidence"
    elif pct >= 50:
        return "moderate confidence — consider expert confirmation"
    return "low confidence — manual inspection recommended"


def get_kb_entry(cls):
    return DISEASE_KB.get(cls, DEFAULT_KB_ENTRY)


def generate_report(pred_class, confidence):
    kb = get_kb_entry(pred_class)
    note = interpret_confidence(confidence)
    readable = pred_class.replace("_", " ")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if openai_key:
        try:
            from openai import OpenAI
            client = OpenAI()
            prompt = (
                f"Explain in under 150 words, for a Bangladeshi farmer, why an AI model predicted "
                f"'{readable}' with {confidence:.1f}% confidence, given typical symptoms "
                f"({kb['symptoms']}). Then give a 'Recommended Treatment' and 'Preventive Measures' "
                f"section using this guidance: treatment='{kb['treatment']}', prevention='{kb['prevention']}'. "
                f"Use simple language and the four headings: Disease Explanation, Confidence Interpretation, "
                f"Recommended Treatment, Preventive Measures."
            )
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3, max_tokens=400,
            )
            return resp.choices[0].message.content, "openai"
        except Exception as e:
            st.warning(f"OpenAI backend unavailable ({e}); using template report.")

    template = f"""**1. Disease Explanation**
The model predicts **{readable}** based on leaf lesions matching: {kb['symptoms']}.

**2. Confidence Interpretation**
The model is {confidence:.1f}% confident ({note}).

**3. Recommended Treatment**
{kb['treatment']}

**4. Preventive Measures**
{kb['prevention']}
"""
    return template, "template"


# =============================== UI ===============================
st.set_page_config(page_title="Bangladeshi Crop Disease Diagnosis", layout="wide")
st.title("🌾 Explainable Crop Disease Diagnosis")
st.caption("Xception classifier + Grad-CAM + SHAP + LLM-based decision support")

if not os.path.exists(MODEL_PATH):
    st.error(f"Trained model not found at `{MODEL_PATH}`. Run the training notebook (Sections 3–4) first.")
    st.stop()

model, idx_to_class = load_model_and_classes()
last_conv_layer_name = find_last_conv_layer(model)

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file)
    raw_img, input_arr = load_and_preprocess(pil_image)

    heatmap, pred_idx, preds = make_gradcam_heatmap(input_arr, model, last_conv_layer_name)
    overlay, heatmap_resized = overlay_gradcam(raw_img, heatmap)

    pred_class = idx_to_class[pred_idx]
    confidence = float(preds[pred_idx] * 100)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(raw_img, caption="Uploaded Leaf Image", use_container_width=True)
    with col2:
        st.image(heatmap_resized, caption="Grad-CAM Heatmap", use_container_width=True, clamp=True)
    with col3:
        st.image(overlay, caption="Grad-CAM Overlay", use_container_width=True)

    st.subheader(f"Prediction: {pred_class.replace('_', ' ')}")
    st.metric("Confidence", f"{confidence:.1f}%")

    with st.spinner("Generating natural-language decision support..."):
        report_text, backend_used = generate_report(pred_class, confidence)

    st.info(f"LLM backend used: **{backend_used}**")
    st.markdown(report_text)

    st.download_button(
        "Download Report (TXT)",
        data=f"Prediction: {pred_class}\nConfidence: {confidence:.1f}%\n\n{report_text}",
        file_name="crop_disease_report.txt",
    )
else:
    st.write("👆 Upload a leaf photo to get started.")
