"""
Streamlit App for Intel Image Classification
Author: Josaiah
Roll No: 125
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import cv2
import os
from pathlib import Path
import io

# Set page configuration
st.set_page_config(
    page_title="Intel Image Classification",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        font-weight: bold;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        font-weight: bold;
        padding: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

class StreamlitImageClassifier:
    """
    Streamlit interface for Intel Image Classification
    """
    
    def __init__(self):
        self.class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
        self.num_classes = len(self.class_names)
        self.img_height = 150
        self.img_width = 150
        
        # Paths
        self.train_dir = 'seg_train/seg_train'
        self.test_dir = 'seg_test/seg_test'
        self.model_path = 'intel_image_classifier_final.h5'
        
    def load_model(self):
        """Load the trained model"""
        try:
            if os.path.exists(self.model_path):
                model = keras.models.load_model(self.model_path)
                return model
            elif os.path.exists('best_model.h5'):
                model = keras.models.load_model('best_model.h5')
                return model
            else:
                return None
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None
    
    def preprocess_image(self, image):
        """Preprocess image for prediction"""
        # Resize image
        img = cv2.resize(image, (self.img_width, self.img_height))
        # Normalize
        img = img / 255.0
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        return img
    
    def predict_image(self, model, image):
        """Make prediction on image"""
        processed_img = self.preprocess_image(image)
        predictions = model.predict(processed_img, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class] * 100
        return predicted_class, confidence, predictions[0]
    
    def display_dataset_info(self):
        """Display dataset information"""
        st.markdown('<p class="sub-header">📊 Dataset Information</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("**Dataset:** Intel Image Classification")
            st.write("**Categories:** 6 classes")
            for idx, class_name in enumerate(self.class_names):
                st.write(f"  {idx}. {class_name.capitalize()}")
        
        with col2:
            st.info("**Image Specifications:**")
            st.write(f"- Image Size: {self.img_height}x{self.img_width} pixels")
            st.write("- Total Training Images: ~14,000")
            st.write("- Total Test Images: ~3,000")
            st.write("- Prediction Images: ~7,000")
    
    def display_sample_images(self):
        """Display sample images from dataset"""
        st.markdown('<p class="sub-header">🖼️ Sample Images from Dataset</p>', unsafe_allow_html=True)
        
        if os.path.exists(self.train_dir):
            cols = st.columns(6)
            
            for idx, (col, class_name) in enumerate(zip(cols, self.class_names)):
                with col:
                    class_path = os.path.join(self.train_dir, class_name)
                    if os.path.exists(class_path):
                        images = [f for f in os.listdir(class_path) 
                                 if f.endswith(('.jpg', '.jpeg', '.png'))]
                        if images:
                            img_path = os.path.join(class_path, np.random.choice(images))
                            img = Image.open(img_path)
                            st.image(img, caption=class_name.capitalize(), use_container_width=True)
        else:
            st.warning("Training directory not found. Please ensure the dataset is in the correct location.")
    
    def display_model_architecture(self):
        """Display model architecture information"""
        st.markdown('<p class="sub-header">🏗️ Model Architecture</p>', unsafe_allow_html=True)
        
        architecture_info = """
        **CNN Architecture Overview:**
        
        1. **Convolutional Blocks (4 blocks):**
           - Block 1: 2x Conv2D(64) + BatchNorm + ReLU + MaxPool + Dropout(0.25)
           - Block 2: 2x Conv2D(128) + BatchNorm + ReLU + MaxPool + Dropout(0.25)
           - Block 3: 2x Conv2D(256) + BatchNorm + ReLU + MaxPool + Dropout(0.25)
           - Block 4: 2x Conv2D(512) + BatchNorm + ReLU + MaxPool + Dropout(0.25)
        
        2. **Fully Connected Layers:**
           - Dense(512) + BatchNorm + ReLU + Dropout(0.5)
           - Dense(256) + BatchNorm + ReLU + Dropout(0.5)
           - Output: Dense(6) + Softmax
        
        3. **Key Features:**
           - ✅ Batch Normalization for stable training
           - ✅ Dropout for regularization
           - ✅ Multiple convolutional layers for feature extraction
           - ✅ Data augmentation during training
        """
        
        st.markdown(architecture_info)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Convolutional Blocks", "4")
        with col2:
            st.metric("Total Conv Layers", "8")
        with col3:
            st.metric("Dense Layers", "3")
    
    def image_prediction_interface(self, model):
        """Image prediction interface"""
        st.markdown('<p class="sub-header">🔮 Image Classification</p>', unsafe_allow_html=True)
        
        # File uploader
        uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
        
        # Option to use sample images
        use_sample = st.checkbox("Or use a sample image from test set")
        
        if use_sample and os.path.exists(self.test_dir):
            selected_class = st.selectbox("Select class:", self.class_names)
            class_path = os.path.join(self.test_dir, selected_class)
            
            if os.path.exists(class_path):
                images = [f for f in os.listdir(class_path) 
                         if f.endswith(('.jpg', '.jpeg', '.png'))]
                if images:
                    selected_image = st.selectbox("Select image:", images[:10])
                    img_path = os.path.join(class_path, selected_image)
                    image = cv2.imread(img_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    uploaded_file = "sample"
                else:
                    st.warning(f"No images found in {selected_class} directory")
                    return
        
        if uploaded_file is not None:
            if uploaded_file != "sample":
                # Read uploaded file
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Display image and prediction
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(image, caption="Input Image", use_container_width=True)
            
            with col2:
                if st.button("🚀 Classify Image", type="primary"):
                    with st.spinner("Analyzing image..."):
                        predicted_class, confidence, all_predictions = self.predict_image(model, image)
                        
                        st.success("Classification Complete!")
                        
                        # Display prediction
                        st.markdown("### Prediction Results")
                        st.metric("Predicted Class", 
                                 self.class_names[predicted_class].capitalize(),
                                 f"{confidence:.2f}% confidence")
                        
                        # Display all class probabilities
                        st.markdown("### Class Probabilities")
                        
                        # Create dataframe for probabilities
                        prob_df = pd.DataFrame({
                            'Class': [name.capitalize() for name in self.class_names],
                            'Probability (%)': all_predictions * 100
                        }).sort_values('Probability (%)', ascending=False)
                        
                        # Display as bar chart
                        fig, ax = plt.subplots(figsize=(10, 4))
                        colors = ['#1f77b4' if i == predicted_class else '#d3d3d3' 
                                 for i in range(len(self.class_names))]
                        ax.barh(prob_df['Class'], prob_df['Probability (%)'], color=colors)
                        ax.set_xlabel('Probability (%)', fontsize=12)
                        ax.set_title('Classification Probabilities', fontsize=14, fontweight='bold')
                        ax.grid(axis='x', alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig)
    
    def display_training_results(self):
        """Display training results if available"""
        st.markdown('<p class="sub-header">📈 Training Results</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        # Check if training history plot exists
        if os.path.exists('training_history.png'):
            with col1:
                st.image('training_history.png', caption='Training History', use_container_width=True)
        else:
            with col1:
                st.info("Training history plot not available. Run training first.")
        
        # Check if confusion matrix exists
        if os.path.exists('confusion_matrix.png'):
            with col2:
                st.image('confusion_matrix.png', caption='Confusion Matrix', use_container_width=True)
        else:
            with col2:
                st.info("Confusion matrix not available. Run evaluation first.")
        
        # Check if sample predictions exist
        if os.path.exists('sample_predictions.png'):
            st.markdown("### Sample Predictions")
            st.image('sample_predictions.png', caption='Sample Predictions', use_container_width=True)


def main():
    """
    Main Streamlit app
    """
    # Header
    st.markdown('<p class="main-header">🖼️ Intel Image Classification with CNN</p>', 
                unsafe_allow_html=True)
    st.markdown("### Lab Assignment 04 - Josaiah (Roll No: 125)")
    st.markdown("---")
    
    # Initialize classifier
    classifier = StreamlitImageClassifier()
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=100)
        st.title("Navigation")
        
        page = st.radio(
            "Select Page:",
            ["🏠 Home", "🔮 Classify Image", "📊 Training Results", "ℹ️ About"]
        )
        
        st.markdown("---")
        st.markdown("### Model Status")
        
        # Check if model exists
        if os.path.exists('intel_image_classifier_final.h5') or os.path.exists('best_model.h5'):
            st.success("✅ Model Available")
        else:
            st.error("❌ Model Not Found")
            st.info("Please train the model first by running `Josaiah_125_Lab04.py`")
    
    # Main content
    if page == "🏠 Home":
        classifier.display_dataset_info()
        st.markdown("---")
        classifier.display_sample_images()
        st.markdown("---")
        classifier.display_model_architecture()
        
    elif page == "🔮 Classify Image":
        model = classifier.load_model()
        if model is not None:
            classifier.image_prediction_interface(model)
        else:
            st.error("❌ Model not found!")
            st.info("Please train the model first by running:")
            st.code("python Josaiah_125_Lab04.py", language="bash")
            
    elif page == "📊 Training Results":
        classifier.display_training_results()
        
    elif page == "ℹ️ About":
        st.markdown('<p class="sub-header">ℹ️ About This Project</p>', unsafe_allow_html=True)
        
        st.markdown("""
        ### Project Overview
        
        This project implements a **Convolutional Neural Network (CNN)** for classifying natural scene images 
        from the Intel Image Classification dataset.
        
        ### Dataset
        - **Source:** Intel Image Classification Dataset
        - **Categories:** 6 classes (buildings, forest, glacier, mountain, sea, street)
        - **Training Images:** ~14,000
        - **Test Images:** ~3,000
        - **Image Size:** 150x150 pixels
        
        ### Model Features
        - ✅ 4 Convolutional Blocks with increasing filter sizes
        - ✅ Batch Normalization for training stability
        - ✅ Dropout layers for regularization
        - ✅ Data Augmentation (rotation, shift, zoom, flip)
        - ✅ Early Stopping and Learning Rate Reduction
        - ✅ Adam Optimizer with categorical cross-entropy loss
        
        ### Implementation Details
        - **Framework:** TensorFlow/Keras
        - **Optimizer:** Adam
        - **Loss Function:** Categorical Cross-Entropy
        - **Training Techniques:** Data Augmentation, Dropout, Batch Normalization
        
        ### Tasks Completed
        1. ✅ Dataset Overview and Visualization
        2. ✅ CNN Model Architecture Design
        3. ✅ Model Training with Callbacks
        4. ✅ Evaluation and Performance Metrics
        5. ✅ Optimization with Data Augmentation
        
        ### Author
        **Name:** Josaiah  
        **Roll No:** 125  
        **Course:** Neural Networks and Deep Learning
        
        ### Technologies Used
        - Python
        - TensorFlow/Keras
        - Streamlit
        - NumPy, Pandas
        - Matplotlib, Seaborn
        - OpenCV
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: gray;'>Intel Image Classification | "
        "Neural Networks Lab Assignment 04 | Josaiah (125)</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
