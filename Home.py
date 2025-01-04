import streamlit as st
from pages import maintenance_assistant, defect_detection, process_monitor

st.set_page_config(page_title="Steel Manufacturing Suite", page_icon="🏭", layout="wide")

# Add custom CSS
st.markdown("""
    <style>
    /* Hide the default Streamlit navigation items */
    header[data-testid="stHeader"] {
        display: none;
    }
    
    div[data-testid="collapsedControl"] {
        display: none;
    }
    
    #MainMenu {
        display: none;
    }
    
    header {
        display: none;
    }
    
    /* Hide Streamlit default pages in sidebar */
    section[data-testid="stSidebar"] > div.element-container:first-child {
        display: none;
    }
    
    /* Rest of your styles */
    .main-header {
        color: #1F618D;
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        margin-bottom: 2rem;
    }
    .feature-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        transition: transform 0.3s ease;
    }
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .stButton>button {
        width: 100%;
        margin-bottom: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)
def show_home():
    st.markdown('<h1 class="main-header">Steel Manufacturing Suite</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
        Welcome to the integrated steel manufacturing monitoring and control system. 
        This suite provides comprehensive tools for maintenance, defect detection, 
        and process monitoring.
    """)
    
    # Feature cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### 🛠️ Maintenance Assistant")
        st.markdown("""
            - AI-powered maintenance support
            - Historical maintenance logs
            - Technical documentation
            - Real-time assistance
        """)
        if st.button("Open Maintenance Assistant", key="btn_maintenance"):
            st.session_state['current_page'] = 'Maintenance'
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### 🔍 Defect Detection")
        st.markdown("""
            - Real-time surface inspection
            - Advanced image processing
            - Defect classification
            - Quality assurance metrics
        """)
        if st.button("Open Defect Detection", key="btn_defect"):
            st.session_state['current_page'] = 'Defect'
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Process Monitor")
        st.markdown("""
            - Real-time parameter tracking
            - Predictive analytics
            - Historical data analysis
            - Performance metrics
        """)
        if st.button("Open Process Monitor", key="btn_process"):
            st.session_state['current_page'] = 'Process'
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 'Home'

# Sidebar navigation
with st.sidebar:
    st.title("Navigation")
    
    # Navigation buttons
    if st.button("🏠 Home", use_container_width=True, type="primary"):
        st.session_state['current_page'] = 'Home'
        st.rerun()
    
    if st.button("🛠️ Maintenance Assistant", use_container_width=True):
        st.session_state['current_page'] = 'Maintenance'
        st.rerun()
    
    if st.button("🔍 Defect Detection", use_container_width=True):
        st.session_state['current_page'] = 'Defect'
        st.rerun()
    
    if st.button("📊 Process Monitor", use_container_width=True):
        st.session_state['current_page'] = 'Process'
        st.rerun()

# Display the selected page
if st.session_state['current_page'] == 'Home':
    show_home()
elif st.session_state['current_page'] == 'Maintenance':
    maintenance_assistant.show_page()
elif st.session_state['current_page'] == 'Defect':
    defect_detection.show_page()
elif st.session_state['current_page'] == 'Process':
    process_monitor.show_page()