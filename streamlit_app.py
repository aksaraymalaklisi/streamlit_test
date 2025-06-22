import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os

# --- Constants for model architecture ---
LATENT_DIM = 100
N_CLASSES = 10
IMG_SIZE = 28
CHANNELS = 1

# --- Define Generator Architecture (must match the trained model) ---
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(N_CLASSES, N_CLASSES)

        self.model = nn.Sequential(
            nn.Linear(LATENT_DIM + N_CLASSES, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, IMG_SIZE * IMG_SIZE * CHANNELS),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), CHANNELS, IMG_SIZE, IMG_SIZE)
        return img

# --- Function to load the model ---
@st.cache_resource
def load_model(model_path):
    """Loads the pre-trained generator model."""
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please train the model first and place it in the 'models' directory.")
        return None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Generator().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# --- Function to generate images ---
def generate_images(generator_model, digit, num_images=5):
    """Generates a specified number of images for a given digit."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Prepare inputs for the generator
    noise = torch.randn(num_images, LATENT_DIM, device=device)
    labels = torch.LongTensor([digit] * num_images).to(device)

    # Generate images
    with torch.no_grad():
        generated_imgs = generator_model(noise, labels)

    # Post-process images for display
    # Rescale from [-1, 1] to [0, 255]
    generated_imgs = (generated_imgs * 0.5 + 0.5) * 255
    generated_imgs = generated_imgs.cpu().numpy().astype(np.uint8)
    
    return generated_imgs

# --- Streamlit App UI ---
st.set_page_config(layout="wide")

st.title("Handwritten Digit Generation with a cGAN")

st.write(
    "This application uses a Conditional Generative Adversarial Network (cGAN) "
    "trained on the MNIST dataset to generate images of handwritten digits. "
    "Select a digit from the dropdown menu below and click the 'Generate' button."
)

# --- Load the generator model ---
model_path = "models/cgan_generator.pth"
generator = load_model(model_path)

if generator:
    # --- User Input ---
    col1, col2 = st.columns([1, 4])
    with col1:
        digit_to_generate = st.selectbox(
            "Select a digit (0-9):",
            list(range(10))
        )
    
    with col2:
        st.write("") # for alignment
        st.write("") # for alignment
        if st.button("Generate Images"):
            st.session_state.generate = True
    
    # --- Display Generated Images ---
    if 'generate' in st.session_state and st.session_state.generate:
        st.subheader(f"Generated Images for Digit: {digit_to_generate}")
        
        # Generate 5 new images
        images = generate_images(generator, digit_to_generate, num_images=5)

        # Display the images in a row
        cols = st.columns(5)
        for i, image_array in enumerate(images):
            with cols[i]:
                st.image(
                    image_array.squeeze(),
                    caption=f"Generated {i+1}",
                    width=150,
                    use_column_width='auto'
                )
else:
    st.warning("Generator model could not be loaded. The application cannot generate images.")

