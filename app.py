import streamlit as st
from plot_functions import plot_confidence_intervals

st.set_page_config(page_title="Confidence Intervals Visualization", layout="wide")

st.title("Confidence Intervals for TPR and TNR")

# Create a layout with controls on the left and plot on the right
col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("Parameters")
    
    # Input parameters
    TPR = st.slider("True Positive Rate (TPR)", 0.0, 1.0, 0.8, 0.01)
    TNR = st.slider("True Negative Rate (TNR)", 0.0, 1.0, 0.8, 0.01)
    prevalence = st.slider("Prevalence", 0.01, 0.99, 0.5, 0.01)
    
    # Sample size controls
    st.subheader("Sample Size")
    Ntot_min = st.number_input("Min Sample Size", 10, 1000, 100, 10)
    Ntot_max = st.number_input("Max Sample Size", 100, 10000, 1000, 100)
    
    # Ensure Ntot_max > Ntot_min
    if Ntot_max <= Ntot_min:
        Ntot_max = Ntot_min + 100
        st.warning("Maximum sample size must be greater than minimum. Adjusted automatically.")
    
    # Confidence level control
    st.subheader("Confidence Settings")
    alpha = st.slider("Significance Level (α)", 0.01, 0.2, 0.05, 0.01)
    confidence_level = (1-alpha)*100
    st.info(f"Confidence Level: {confidence_level:.0f}%")

with col2:
    # Generate the plot with the specified parameters
    try:
        fig = plot_confidence_intervals(TPR, TNR, prevalence, Ntot_min, Ntot_max, alpha)
        st.pyplot(fig)  # Use st.pyplot for matplotlib figures
    except Exception as e:
        st.error(f"Error generating plot: {str(e)}")
    
    # Add explanation
    st.markdown("""
    ### About this Visualization
    
    This plot shows how confidence intervals for True Positive Rate (TPR) and True Negative Rate (TNR) 
    change with increasing sample size.
    
    - **Top row**: TPR confidence intervals (blue, left) and TNR confidence intervals (red, right)
    - **Bottom row**: Width of confidence intervals for both metrics
    
    The solid horizontal lines represent the true TPR and TNR values. As sample size increases, 
    confidence intervals narrow, indicating increased precision in our estimates.
    """) 