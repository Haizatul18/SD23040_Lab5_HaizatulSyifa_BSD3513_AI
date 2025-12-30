# File: SD23040_Lab5_CV.py
import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

# Configure page
st.set_page_config(page_title="Image Classification App", layout="centered")
st.title("CPU-based Image Classification with ResNet18")

# Force CPU usage
device = torch.device('cpu')

# Load pre-trained ResNet18
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()
model.to(device)

# Preprocessing transformations
preprocess = models.ResNet18_Weights.DEFAULT.transforms()

# Upload image
uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Apply transformations and add batch dimension
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    # Model inference (no gradient)
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Get top 5 predictions
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    classes = [models.ResNet18_Weights.DEFAULT.meta["categories"][i] for i in top5_catid]

    # Display predictions
    st.subheader("Top-5 Predictions:")
    for i in range(5):
        st.write(f"{classes[i]}: {top5_prob[i].item():.4f}")

    # Visualize probabilities as a bar chart
    df = pd.DataFrame({'Class': classes, 'Probability': top5_prob.numpy()})
    st.bar_chart(df.set_index('Class'))
