import gradio as gr
import tensorflow as tf
import numpy as np
import json
import cv2
import tempfile
import os
from PIL import Image
from datetime import datetime

MODEL_PATH = "mobilenetv2_model.keras"
CLASS_PATH = "class_names.json"

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_PATH) as f:
    class_names = json.load(f)

base = model.get_layer("mobilenetv2_1.00_224")

feature_model = tf.keras.Model(
    base.input,
    base.get_layer("out_relu").output
)

inputs = tf.keras.Input(shape=(224, 224, 3))
x = inputs

for layer in model.layers[1:3]:
    x = layer(x, training=False)

features = feature_model(x)
x = features

for layer in model.layers[4:]:
    x = layer(x)

grad_model = tf.keras.Model(inputs, [features, x])


def disease_information(disease):
    name = disease.lower()

    if "cercospora" in name or "gray_leaf_spot" in name:
        return (
            "Gray Leaf Spot",
            "Small gray, tan or brown lesions may develop on leaves and enlarge over time.",
            "Use resistant varieties, remove infected crop residue, improve field airflow "
            "and follow locally recommended disease management practices."
        )

    if "common_rust" in name:
        return (
            "Common Rust",
            "Rust-colored pustules may appear on leaf surfaces.",
            "Use resistant varieties, monitor crops regularly and follow local agricultural "
            "recommendations for disease management."
        )

    if "northern_leaf_blight" in name:
        return (
            "Northern Leaf Blight",
            "Long, elongated gray-green or tan lesions may develop on leaves.",
            "Use resistant varieties, manage crop residue and follow recommended crop rotation "
            "and disease management practices."
        )

    if "healthy" in name:
        return (
            "Healthy Plant",
            "The model classified the leaf as healthy.",
            "Continue regular crop monitoring, proper irrigation and balanced nutrition."
        )

    return (
        disease.replace("___", " - ").replace("_", " ").title(),
        "The model detected visual patterns associated with this disease class.",
        "Monitor the plant, remove severely affected material where appropriate, "
        "avoid unnecessary leaf wetness and consult local agricultural guidance."
    )


def detect_disease_spots(img):
    image = np.array(img.convert("RGB").resize((224, 224)))
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    h, w = image.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)

    cv2.grabCut(
        bgr, mask, (5, 5, w-10, h-10),
        bgd, fgd, 5, cv2.GC_INIT_WITH_RECT
    )

    leaf_mask = np.where(
        (mask == 1) | (mask == 3), 255, 0
    ).astype(np.uint8)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        leaf_mask, 8
    )

    if count > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        leaf_mask = np.where(
            labels == largest, 255, 0
        ).astype(np.uint8)

    inner_mask = cv2.erode(
        leaf_mask, np.ones((7, 7), np.uint8), iterations=1
    )

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h_channel, s, v = cv2.split(hsv)

    lesions = (
        (h_channel >= 3) & (h_channel <= 25) &
        (s > 45) & (v > 40) & (v < 220) &
        (inner_mask > 0)
    )

    disease_mask = lesions.astype(np.uint8) * 255

    disease_mask = cv2.morphologyEx(
        disease_mask, cv2.MORPH_OPEN,
        np.ones((3, 3), np.uint8)
    )

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        disease_mask, 8
    )

    clean_mask = np.zeros_like(disease_mask)

    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] >= 12:
            clean_mask[labels == i] = 255

    clean_mask = cv2.bitwise_and(clean_mask, inner_mask)

    leaf_pixels = np.count_nonzero(inner_mask)
    disease_pixels = np.count_nonzero(clean_mask)
    severity = disease_pixels / max(leaf_pixels, 1) * 100

    overlay = image.copy()
    overlay[clean_mask > 0] = (
        0.6 * image[clean_mask > 0] +
        0.4 * np.array([255, 0, 0])
    ).astype(np.uint8)

    return overlay, severity


def predict(img):
    if img is None:
        raise gr.Error("Please upload a plant leaf image.")

    img = img.convert("RGB").resize((224, 224))
    original = np.array(img, dtype=np.float32)

    array = np.expand_dims(original.copy(), axis=0)
    array = tf.keras.applications.mobilenet_v2.preprocess_input(array)

    with tf.GradientTape() as tape:
        features, predictions = grad_model(array, training=False)
        index = tf.argmax(predictions[0])
        loss = predictions[:, index]

    gradients = tape.gradient(loss, features)
    weights = tf.reduce_mean(gradients, axis=(1, 2))

    heatmap = tf.reduce_sum(
        features[0] * weights[0], axis=-1
    )

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8
    heatmap = heatmap.numpy()

    prediction = predictions[0].numpy()
    top_indices = np.argsort(prediction)[-3:][::-1]

    disease = class_names[int(index)]
    confidence = float(prediction[int(index)]) * 100

    top3 = "\n".join(
        f"{class_names[i]}: {prediction[i]*100:.2f}%"
        for i in top_indices
    )

    heatmap_resized = cv2.resize(heatmap, (224, 224))

    colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized),
        cv2.COLORMAP_JET
    )

    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    gradcam_overlay = cv2.addWeighted(
        original.astype(np.uint8),
        0.6,
        colored,
        0.4,
        0
    )

    spot_overlay, severity = detect_disease_spots(img)

    display_name, symptoms, prevention = disease_information(disease)

    report = f"""
AGRIVISION-AI
PLANT DISEASE ANALYSIS REPORT
================================

Date: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}

PREDICTION
--------------------------------
Predicted Disease: {disease}
Disease Name: {display_name}
Model Confidence: {confidence:.2f}%

TOP 3 PREDICTIONS
--------------------------------
{top3}

DISEASE INFORMATION
--------------------------------
Symptoms:
{symptoms}

Prevention:
{prevention}

DISEASE SPOT ANALYSIS
--------------------------------
Estimated Affected Leaf Area: {severity:.2f}%

Note:
Affected area is an approximate color-based image estimate.
It is not a validated measurement of disease severity.
AI predictions should be verified with appropriate agricultural expertise.
"""

    fd, report_path = tempfile.mkstemp(
        suffix=".txt",
        prefix="AgriVision_Report_"
    )

    with os.fdopen(fd, "w") as file:
        file.write(report)

    return (
        disease,
        f"{confidence:.2f}%",
        top3,
        heatmap_resized,
        gradcam_overlay,
        spot_overlay,
        f"{severity:.2f}%",
        display_name,
        symptoms,
        prevention,
        report_path
    )


custom_css = """
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
}
#main-title {
    text-align: center;
    padding: 22px;
    background: linear-gradient(120deg, #14532d, #16a34a);
    border-radius: 15px;
    color: white;
    margin-bottom: 15px;
}
#main-title h1 {
    color: white !important;
    font-size: 36px !important;
}
#main-title p {
    color: #ecfdf5 !important;
    font-size: 16px;
}
.section-title {
    color: #15803d;
    font-weight: bold;
}
"""

with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="green",
        secondary_hue="emerald"
    ),
    css=custom_css,
    title="AgriVision-AI Dashboard"
) as demo:

    gr.HTML("""
    <div id="main-title">
        <h1>🌿 AgriVision-AI</h1>
        <p>Intelligent Plant Disease Detection and Analysis System</p>
        <p>AI-Powered Agriculture | MobileNetV2 | Grad-CAM</p>
    </div>
    """)

    gr.Markdown(
        "### Welcome to your Plant Health Dashboard\n"
        "Upload a clear leaf image to analyze its possible disease."
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil",
                label="Upload Plant Leaf",
                height=300
            )

            detect_button = gr.Button(
                "🔍 Analyze Plant",
                variant="primary",
                size="lg"
            )

        with gr.Column(scale=1):
            disease_output = gr.Textbox(
                label="Predicted Disease",
                interactive=False
            )

            confidence_output = gr.Textbox(
                label="Model Confidence",
                interactive=False
            )

            top3_output = gr.Textbox(
                label="Top 3 Predictions",
                lines=4,
                interactive=False
            )

    gr.Markdown("## 📊 Disease Visualization")

    with gr.Row():
        heatmap_output = gr.Image(
            label="Grad-CAM Heatmap"
        )

        gradcam_output = gr.Image(
            label="Disease Attention Overlay"
        )

    gr.Markdown("## 🔴 Disease Spot Detection")

    with gr.Row():
        spot_output = gr.Image(
            label="Detected Spots in Red"
        )

        severity_output = gr.Textbox(
            label="Estimated Affected Leaf Area",
            interactive=False
        )

    gr.Markdown("## 🌱 Disease Information")

    with gr.Row():
        display_name_output = gr.Textbox(
            label="Disease Name",
            interactive=False
        )

    symptoms_output = gr.Textbox(
        label="Possible Symptoms",
        lines=3,
        interactive=False
    )

    prevention_output = gr.Textbox(
        label="Prevention and Management Suggestions",
        lines=4,
        interactive=False
    )

    gr.Markdown("## 📄 Download Analysis Report")

    report_output = gr.File(
        label="Your Disease Analysis Report"
    )

    detect_button.click(
        fn=predict,
        inputs=image_input,
        outputs=[
            disease_output,
            confidence_output,
            top3_output,
            heatmap_output,
            gradcam_output,
            spot_output,
            severity_output,
            display_name_output,
            symptoms_output,
            prevention_output,
            report_output
        ]
    )

    gr.Markdown("""
    ---
    <center>
    **AgriVision-AI | Plant Disease Detection System**

    Developed using Python, TensorFlow, MobileNetV2, OpenCV and Gradio.

    *AI predictions and affected-area estimates are approximate and should be verified.*
    </center>
    """)

demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))