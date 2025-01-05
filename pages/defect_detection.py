import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import os

def load_css():
    """Load custom CSS styles"""
    st.markdown("""
        <style>
        .main-header {
            color: #1F618D;
            padding: 1rem;
            background-color: #f8f9fa;
            border-radius: 0.5rem;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: white;
            padding: 1.5rem;
            border-radius: 0.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        }
        .upload-section {
            border: 2px dashed #cccccc;
            border-radius: 0.5rem;
            padding: 2rem;
            text-align: center;
            background-color: #f8f9fa;
        }
        .status-defect {
            background-color: #fff3cd;
            color: #856404;
            padding: 1rem;
            border-radius: 0.5rem;
            border: 1px solid #ffeeba;
            margin: 1rem 0;
        }
        .status-ok {
            background-color: #d4edda;
            color: #155724;
            padding: 1rem;
            border-radius: 0.5rem;
            border: 1px solid #c3e6cb;
            margin: 1rem 0;
        }
        .probability-bar {
            margin: 0.5rem 0;
            padding: 0.5rem;
            background-color: #f8f9fa;
            border-radius: 0.25rem;
        }
        .model-info {
            background-color: #e3f2fd;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        .error-message {
            background-color: #f8d7da;
            color: #721c24;
            padding: 1rem;
            border-radius: 0.5rem;
            border: 1px solid #f5c6cb;
            margin: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)

class DefectClassifier(nn.Module):
    """ResNet18-based defect classifier"""
    def __init__(self, num_classes=6):
        super(DefectClassifier, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)

@st.cache_resource
def load_model():
    """Load and cache the model with error handling"""
    try:
        model = DefectClassifier(num_classes=6)
        if not os.path.exists('models/best_model.pth'):
            raise FileNotFoundError("Model file not found. Please ensure 'models/best_model.pth' exists.")
        
        checkpoint = torch.load('models/best_model.pth', map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model, None
    except Exception as e:
        return None, str(e)

def preprocess_image(image):
    """Preprocess image for model input"""
    try:
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        return transform(image).unsqueeze(0), None
    except Exception as e:
        return None, str(e)

def get_prediction(model, image):
    """Get model predictions with confidence scores"""
    classes = [
        'Crazing', 'Inclusion', 'Patches', 
        'Pitted Surface', 'Rolled-in Scale', 'Scratches'
    ]
    try:
        with torch.no_grad():
            outputs = model(image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            predictions = {
                classes[i]: float(probabilities[i]) 
                for i in range(len(classes))
            }
            return predictions, None
    except Exception as e:
        return None, str(e)

def create_gauge_chart(value, title):
    """Create a gauge chart for confidence visualization"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        title={'text': title, 'font': {'size': 24}},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 75], 'color': "gray"},
                {'range': [75, 100], 'color': "darkgray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(
        height=200,
        margin=dict(l=10, r=10, t=30, b=10),
        font={'size': 16}
    )
    return fig

def display_model_info():
    """Display model information and settings"""
    st.markdown("""
        <div class="model-info">
            <h4>Model Information</h4>
            <p>• Architecture: ResNet18</p>
            <p>• Input Size: 224x224 pixels</p>
            <p>• Classes: 6 defect types</p>
            <p>• Framework: PyTorch</p>
        </div>
    """, unsafe_allow_html=True)

def show_page():
    """Display the defect detection page"""
    load_css()
    
    # Page title
    st.markdown(
        '<h1 class="main-header">Steel Surface Defect Detection</h1>', 
        unsafe_allow_html=True
    )
    
    # Load model
    model, model_error = load_model()
    if model_error:
        st.markdown(f"""
            <div class="error-message">
                <h4>❌ Error Loading Model</h4>
                <p>{model_error}</p>
            </div>
        """, unsafe_allow_html=True)
        return
    
    st.success("✅ Model loaded successfully!")
    
    # Initialize session state for image
    if 'defect_image' not in st.session_state:
        st.session_state.defect_image = None
    
    # Sidebar settings
    with st.sidebar:
        st.markdown('<h2 class="main-header">Detection Settings</h2>', unsafe_allow_html=True)
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Minimum confidence level to consider a defect detection valid"
        )
    
    display_model_info()
    
    # Main layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload Steel Surface Image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image of the steel surface for defect detection"
        )
        
        if uploaded_file:
            try:
                st.session_state.defect_image = Image.open(uploaded_file)
                st.image(st.session_state.defect_image, caption="Uploaded Image", use_column_width=True)
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")
        else:
            st.markdown("### 📤 Upload an image to begin analysis")
        st.markdown("</div></div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        if st.session_state.defect_image is not None:
            try:
                # Process image
                processed_image, preprocess_error = preprocess_image(st.session_state.defect_image)
                if preprocess_error:
                    st.error(f"Error preprocessing image: {preprocess_error}")
                    return
                
                # Get predictions
                predictions, prediction_error = get_prediction(model, processed_image)
                if prediction_error:
                    st.error(f"Error getting predictions: {prediction_error}")
                    return
                
                # Get highest probability defect
                max_defect = max(predictions.items(), key=lambda x: x[1])
                
                # Display results
                st.markdown("### Analysis Results")
                
                if max_defect[1] > confidence_threshold:
                    st.markdown(
                        f"""<p class='status-defect'>
                            ⚠ Defect Detected: {max_defect[0]}
                            <br>Confidence: {max_defect[1]*100:.1f}%
                        </p>""", 
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""<p class='status-ok'>
                            ✅ No Significant Defects
                            <br>Highest probability: {max_defect[1]*100:.1f}%
                        </p>""", 
                        unsafe_allow_html=True
                    )
                
                # Show confidence gauge
                fig = create_gauge_chart(max_defect[1], "Confidence Level")
                st.plotly_chart(fig, use_container_width=True)
                
                # Show all probabilities
                st.markdown("### Defect Probabilities")
                sorted_predictions = dict(sorted(predictions.items(), 
                                              key=lambda x: x[1], 
                                              reverse=True))
                for defect, prob in sorted_predictions.items():
                    st.markdown("<div class='probability-bar'>", unsafe_allow_html=True)
                    st.progress(prob)
                    st.markdown(f"**{defect}**: {prob*100:.1f}%")
                    st.markdown("</div>", unsafe_allow_html=True)
                
            except Exception as e:
                st.markdown(f"""
                    <div class="error-message">
                        <h4>❌ Error Processing Image</h4>
                        <p>{str(e)}</p>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("### Upload an image to see analysis results")
        
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    st.set_page_config(
        page_title="Steel Defect Detection",
        page_icon="🔍",
        layout="wide"
    )
    show_page()