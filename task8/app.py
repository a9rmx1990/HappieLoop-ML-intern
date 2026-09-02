import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage
import os

st.set_page_config(page_title="Task 8: PyTorch Neural Network", page_icon="🧠", layout="wide")

st.title("🧠 Task 8: PyTorch Handwritten Digit Classifier (MNIST)")
st.markdown("Interactive Deep Learning Dashboard with **Smart Center-of-Mass Preprocessing** for custom drawings.")

# Resolve paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_PATH = os.path.join(BASE_DIR, "mnist_model.pth")

# Define Model Architecture
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

@st.cache_resource
def load_trained_model():
    model = SimpleNN()
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=True))
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = MNIST(root=DATA_DIR, train=True, download=False, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        for _ in range(3):
            for imgs, lbls in train_loader:
                optimizer.zero_grad()
                out = model(imgs)
                loss = criterion(out, lbls)
                loss.backward()
                optimizer.step()
        torch.save(model.state_dict(), MODEL_PATH)
        
    model.eval()
    return model

@st.cache_data
def get_test_samples():
    transform = transforms.Compose([transforms.ToTensor()])
    test_ds = MNIST(root=DATA_DIR, train=False, download=False, transform=transform)
    return test_ds

with st.spinner("Loading PyTorch model..."):
    model = load_trained_model()
    test_ds = get_test_samples()

# True MNIST-Standard Bounding-Box & Center-of-Mass Extraction
def preprocess_custom_image(pil_img, invert_colors=True, threshold_val=100):
    """
    Standard MNIST Preprocessing Pipeline:
    1. Grayscale conversion + contrast enhancement
    2. Adaptive thresholding to remove background noise (no blur)
    3. Bounding box cropping around the exact digit strokes
    4. Aspect-ratio preserving scaling into a 20x20 box
    5. Centering inside a 28x28 canvas using Center of Mass (how MNIST was created)
    """
    # 1. Grayscale
    gray = pil_img.convert("L")
    if invert_colors:
        gray = ImageOps.invert(gray)
        
    # 2. Contrast thresholding to make strokes crisp without blur
    arr = np.array(gray, dtype=np.float32)
    # Remove faint background noise below threshold
    arr[arr < threshold_val] = 0.0
    
    # 3. Find Bounding Box of the drawn digit
    nonzero_coords = np.argwhere(arr > 0)
    if len(nonzero_coords) == 0:
        return np.zeros((28, 28), dtype=np.float32)
        
    ymin, xmin = nonzero_coords.min(axis=0)
    ymax, xmax = nonzero_coords.max(axis=0) + 1
    
    cropped = arr[ymin:ymax, xmin:xmax]
    crop_h, crop_w = cropped.shape
    
    # 4. Scale preserving aspect ratio to fit inside 20x20
    crop_img = Image.fromarray(cropped.astype(np.uint8))
    if crop_h > crop_w:
        new_h = 20
        new_w = max(1, int(round((crop_w / crop_h) * 20.0)))
    else:
        new_w = 20
        new_h = max(1, int(round((crop_h / crop_w) * 20.0)))
        
    resized_digit = crop_img.resize((new_w, new_h), Image.Resampling.BICUBIC)
    resized_arr = np.array(resized_digit, dtype=np.float32)
    
    # 5. Place in 28x28 canvas centered by Center of Mass
    canvas = np.zeros((28, 28), dtype=np.float32)
    start_y = (28 - new_h) // 2
    start_x = (28 - new_w) // 2
    canvas[start_y:start_y + new_h, start_x:start_x + new_w] = resized_arr
    
    # Adjust center of mass shift to match MNIST centering
    cy, cx = ndimage.center_of_mass(canvas)
    shift_y = int(round(14.0 - cy))
    shift_x = int(round(14.0 - cx))
    
    canvas = ndimage.shift(canvas, [shift_y, shift_x], mode='constant', cval=0.0)
    canvas = np.clip(canvas / 255.0, 0.0, 1.0)
    
    return canvas

# Sidebar Mode Selection
st.sidebar.header("Input Mode")
mode = st.sidebar.radio("Choose Input Method:", ["📁 Browse MNIST Test Set", "📤 Upload Your Own Digit Image"])

norm_transform = transforms.Normalize((0.1307,), (0.3081,))

if mode == "📁 Browse MNIST Test Set":
    sample_index = st.sidebar.slider("Select Test Sample Index", min_value=0, max_value=len(test_ds)-1, value=42)
    sample_img_tensor, true_label = test_ds[sample_index]
    
    input_tensor = norm_transform(sample_img_tensor).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.nn.functional.softmax(logits[0], dim=0).numpy()
        pred_label = int(np.argmax(probs))
        conf = float(probs[pred_label])

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader(f"Selected Digit (Index #{sample_index})")
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(sample_img_tensor.squeeze(), cmap="gray")
        ax.axis("off")
        st.pyplot(fig)
        
        st.markdown(f"**Ground Truth Label:** `{true_label}`")
        status_color = "green" if pred_label == true_label else "red"
        st.markdown(f"**Model Prediction:** <span style='color:{status_color}; font-size:1.4rem; font-weight:bold;'>{pred_label}</span>", unsafe_allow_html=True)
        st.markdown(f"**Confidence:** `{conf * 100:.2f}%`")

    with col2:
        st.subheader("Model Output Class Probabilities")
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(range(10), probs * 100, color=['#4361ee' if i != pred_label else '#38b000' for i in range(10)])
        ax.set_xticks(range(10))
        ax.set_xlabel("Digit Class (0-9)")
        ax.set_ylabel("Probability (%)")
        ax.set_ylim(0, 105)
        for bar in bars:
            height = bar.get_height()
            if height > 1:
                ax.annotate(f'{height:.1f}%',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)
        st.pyplot(fig)

else:
    st.subheader("📤 Upload a Custom Digit Image")
    st.markdown("""
    Upload any picture or drawing of a single digit (0–9). 
    The app uses **Bounding-Box Cropping & Center-of-Mass Alignment** to preserve sharp stroke thickness and position the digit exactly as the model expects.
    """)
    
    uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])
    
    st.sidebar.subheader("Image Extraction Settings")
    invert_colors = st.sidebar.checkbox("Invert Colors (Dark ink on Light paper)", value=True)
    threshold_val = st.sidebar.slider("Stroke Contrast Threshold", min_value=10, max_value=200, value=80,
                                      help="Removes faint background noise to keep line strokes clean and sharp.")

    if uploaded_file is not None:
        user_img = Image.open(uploaded_file)
        
        # Apply smart bounding-box & center-of-mass extraction
        preprocessed_arr = preprocess_custom_image(user_img, invert_colors=invert_colors, threshold_val=threshold_val)
        
        # Convert to PyTorch Tensor
        img_tensor = torch.tensor(preprocessed_arr, dtype=torch.float32).unsqueeze(0)  # shape: (1, 28, 28)
        input_tensor = norm_transform(img_tensor).unsqueeze(0)  # shape: (1, 1, 28, 28)
        
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.nn.functional.softmax(logits[0], dim=0).numpy()
            pred_label = int(np.argmax(probs))
            conf = float(probs[pred_label])
            
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            st.subheader("Original Upload")
            st.image(user_img, use_container_width=True)

        with col2:
            st.subheader("Extracted & Centered")
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.imshow(preprocessed_arr, cmap="gray")
            ax.axis("off")
            st.pyplot(fig)
            
            st.markdown(f"**Predicted:** <span style='color:#38b000; font-size:1.8rem; font-weight:bold;'>{pred_label}</span>", unsafe_allow_html=True)
            st.markdown(f"**Confidence:** `{conf * 100:.2f}%`")
            
        with col3:
            st.subheader("Model Class Probabilities")
            fig, ax = plt.subplots(figsize=(7, 4))
            bars = ax.bar(range(10), probs * 100, color=['#4361ee' if i != pred_label else '#38b000' for i in range(10)])
            ax.set_xticks(range(10))
            ax.set_xlabel("Digit Class (0-9)")
            ax.set_ylabel("Probability (%)")
            ax.set_ylim(0, 105)
            for bar in bars:
                height = bar.get_height()
                if height > 1:
                    ax.annotate(f'{height:.1f}%',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3), textcoords="offset points",
                                ha='center', va='bottom', fontsize=8)
            st.pyplot(fig)
    else:
        st.info("👆 Upload an image of a handwritten digit above to test the model!")

st.divider()
st.subheader("🏗️ Model Architecture Details")
st.code("""
SimpleNN(
  (net): Sequential(
    (0): Flatten(start_dim=1, end_dim=-1)
    (1): Linear(in_features=784, out_features=256, bias=True)
    (2): ReLU()
    (3): Dropout(p=0.3, inplace=False)
    (4): Linear(in_features=256, out_features=128, bias=True)
    (5): ReLU()
    (6): Linear(in_features=128, out_features=10, bias=True)
  )
)
- Total Parameters: 235,146
- Optimizer: Adam (lr=1e-3)
- Regularization: Dropout (p=0.3) & StepLR decay
""", language="text")
