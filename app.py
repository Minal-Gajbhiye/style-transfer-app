import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
from io import BytesIO

# --- Debugging Message ---
st.write("1. Script execution started.")

# --- Model and Image Processing Logic ---

# @st.cache_resource tells Streamlit to run this function only once.
@st.cache_resource
def load_style_transfer_model():
    """Loads the fast style transfer model from TensorFlow Hub."""
    st.write("2. Cache miss: Loading style transfer model from TensorFlow Hub for the first time...")
    model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
    st.write("3. Model loaded successfully.")
    return model

def tensor_to_image(tensor):
    """Converts a tensor to a PIL Image."""
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)

def process_image(image_file, target_dim=512):
    """Loads and preprocesses an image from an upload."""
    img = Image.open(image_file)
    img = tf.keras.utils.img_to_array(img)
    img = tf.image.convert_image_dtype(img, tf.float32)
    
    # Resize the image to the target dimension
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = target_dim / long_dim
    new_shape = tf.cast(shape * scale, tf.int32)
    
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

# --- Streamlit Web App Interface ---

st.set_page_config(page_title="Fast Artistic Style Transfer", layout="wide")
st.title("🎨 Fast Artistic Style Transfer")
st.write("This version uses an optimized model from TensorFlow Hub for much faster results.")

st.sidebar.header("Instructions")
st.sidebar.info("Upload a content image and a style image, then click 'Generate' to see the magic happen almost instantly.")

content_file = st.sidebar.file_uploader("1. Upload Content Image", type=["jpg", "jpeg", "png"])
style_file = st.sidebar.file_uploader("2. Upload Style Image", type=["jpg", "jpeg", "png"])

# Display uploaded images
col1, col2 = st.columns(2)
if content_file:
    col1.header("Content Image")
    col1.image(content_file, use_column_width=True)

if style_file:
    col2.header("Style Image")
    col2.image(style_file, use_column_width=True)

if content_file and style_file:
    st.write("4. Both files have been uploaded.")
    if st.sidebar.button("Generate Artistic Image", use_container_width=True, type="primary"):
        st.write("5. Generate button clicked.")
        # Load the model
        hub_model = load_style_transfer_model()

        with st.spinner("Stylizing..."):
            st.write("6. Processing images...")
            # Process images
            content_image = process_image(content_file)
            style_image = process_image(style_file)

            st.write("7. Performing style transfer...")
            # Perform style transfer
            stylized_image_tensor = hub_model(tf.constant(content_image), tf.constant(style_image))[0]
            
            st.write("8. Converting tensor to image...")
            # Convert tensor to image
            final_image = tensor_to_image(stylized_image_tensor)

            st.header("Generated Artwork")
            st.image(final_image, caption="Your stylized artwork.", use_column_width=True)
            
            # Prepare for download
            buf = BytesIO()
            final_image.save(buf, format="PNG")
            byte_im = buf.getvalue()
            
            st.download_button(
                label="Download Image",
                data=byte_im,
                file_name="styled_image.png",
                mime="image/png",
                use_container_width=True
            )
st.write("9. Script execution finished.")