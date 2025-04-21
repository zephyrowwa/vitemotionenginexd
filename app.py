import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from timm import create_model

class_labels = ['angry', 'happy', 'neutral', 'sad']

@st.cache_resource
def load_model():
    model = create_model('vit_base_patch16_224', pretrained=False, num_classes=4)
    model.load_state_dict(torch.load("vit for streamlit.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    return transform(image).unsqueeze(0) 

def predict(model, input_tensor):
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)
        _, predicted = torch.max(probabilities, 1)
        return class_labels[predicted.item()], probabilities.squeeze().tolist()

st.set_page_config(page_title="Facial Emotion Recognition", layout="centered")
st.title("Facial Emotion Recognition")
st.write("Upload an image to classify the emotion using a Vision Transformer (ViT) model.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    model = load_model()
    input_tensor = preprocess_image(image)
    label, probs = predict(model, input_tensor)

    st.success(f"Predicted Emotion: **{label}**")

    st.subheader("Confidence Scores")
    prob_dict = {label: prob for label, prob in zip(class_labels, probs)}
    st.bar_chart(prob_dict)
