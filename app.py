import streamlit as st
import joblib
import numpy as np
import time

# Set the page configuration
# This must be the first Streamlit command
st.set_page_config(
    page_title="Simple Salary Predictor",
    page_icon="💵",
    layout="centered",
    initial_sidebar_state="auto"
)

# Custom CSS to inject
st.markdown("""
<style>
    /* Main app layout */
    .main .block-container {
        max-width: 600px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Card effect */
    .card {
        background-color: var(--secondary-background-color);
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    /* Title */
    h1 {
        text-align: center;
        color: var(--primary-color);
    }
    
    /* Subtitle */
    .subtitle {
        text-align: center;
        font-size: 1.1rem;
        color: var(--text-color);
        opacity: 0.8;
        margin-bottom: 2rem;
    }
    
    /* Styled Button */
    .stButton > button {
        width: 100%;
        height: 3.5rem;
        font-size: 1.25rem;
        font-weight: bold;
        border-radius: 8px;
        background-image: linear-gradient(to right, #00c6ff 0%, #0072ff 51%, #00c6ff 100%);
        color: white;
        border: none;
        transition: 0.5s;
        background-size: 200% auto;
    }
    .stButton > button:hover {
        background-position: right center;
    }
    
    /* Result Box */
    .result-box {
        background-color: #0072ff22;
        border: 2px solid #0072ff;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        margin-top: 2rem;
    }
    .result-label {
        font-size: 1.1rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    .result-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: var(--primary-color);
    }
</style>
""", unsafe_allow_html=True)


# --- App ---

# Load the trained model
# Use st.cache_resource to load the model only once
@st.cache_resource
def load_model():
    """Loads the simple 'salary_model.pkl' file."""
    try:
        model = joblib.load("salary_model.pkl")
        return model
    except FileNotFoundError:
        st.error("Model file 'salary_model.pkl' not found. Please run `train.py` first.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None

model = load_model()

if model:
    # Add a title and subtitle
    st.title("Simple Salary Predictor 💵")
    st.markdown("<p class='subtitle'>Predict a salary based on years of experience.</p>", unsafe_allow_html=True)

    # Wrap the main content in a card
    st.markdown('<div class="card">', unsafe_allow_html=True)

    # Create a number input widget
    years = st.number_input(
        "Enter Years of Experience:",
        min_value=0.0,
        max_value=50.0,
        step=0.5,
        value=1.0
    )

    # Create a prediction button
    if st.button("Predict Salary"):
        with st.spinner('Calculating...'):
            time.sleep(0.5) # Simulate a short delay
            
            # Convert the input to a numpy array for the model
            features = np.array([[years]])
            
            # Make a prediction
            prediction = model.predict(features)
            
            # Display the result in the custom box
            st.markdown(f"""
            <div class="result-box">
                <div class="result-label">Predicted Annual Salary:</div>
                <div class="result-value">${prediction[0]:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)

    # Close the card div
    st.markdown('</div>', unsafe_allow_html=True)

    # Add an expander for "How it works"
    with st.expander("How it works", expanded=False):
        st.write("""
            This app uses a **Simple Linear Regression** model.
            

[Image of a 2D scatter plot with a linear regression line]

            
            1.  The model was trained on a public dataset (`Salary_Data.csv`) of salaries and years of experience.
            2.  When you enter a number, the model (loaded in `salary_model.pkl`) predicts the most likely salary.
        """)