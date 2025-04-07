import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(page_title="Ad Spend ROAS Calculator", layout="wide")

# App title
st.title("Ad Spend ROAS Calculator")
st.markdown("Simulate the effect of changing ad spend on ROAS using a diminishing returns model")

# Sidebar for data upload
with st.sidebar:
    st.header("Upload Data")
    uploaded_file = st.file_uploader("Choose a CSV file with Spend and ROAS columns", type="csv")
    
    model_type = st.selectbox("Select Model Type", ["Logarithmic", "Polynomial"])
    caution_threshold = st.slider("Caution Threshold (%)", 1, 20, 10)

# Main functionality
if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    
    # Display data preview
    st.subheader("Data Preview")
    st.dataframe(df.head())
    
    # Check for required columns
    if 'Spend' not in df.columns or 'ROAS' not in df.columns:
        st.error("CSV must contain 'Spend' and 'ROAS' columns")
    else:
        # Fit model
        spend = df['Spend'].values
        roas = df['ROAS'].values
        
        if model_type == "Logarithmic":
            # Remove zero spend values for log model
            valid_idx = spend > 0
            spend_valid = spend[valid_idx]
            roas_valid = roas[valid_idx]
            
            # Fit log model: ROAS = a + b*log(spend)
            coeffs = np.polyfit(np.log(spend_valid), roas_valid, 1)
            b, a = coeffs
            
            def predict(x):
                return a + b * np.log(x)
                
            equation = f"ROAS = {a:.4f} + {b:.4f} × log(Spend)"
        else:
            # Fit polynomial: ROAS = a + b*spend + c*spend^2
            coeffs = np.polyfit(spend, roas, 2)
            c, b, a = coeffs
            
            def predict(x):
                return a + b * x + c * x * x
                
            equation = f"ROAS = {a:.4f} + {b:.4f} × Spend + {c:.4f} × Spend²"
        
        # User inputs
        st.subheader("Simulator")
        col1, col2 = st.columns(2)
        with col1:
            current_spend = st.number_input("Current Daily Ad Spend ($)", 
                                           min_value=0.01, value=float(spend.mean()))
        with col2:
            new_spend = st.number_input("Proposed Daily Ad Spend ($)", 
                                       min_value=0.01, value=float(spend.mean() * 1.2))
        
        # Calculate results
        current_roas = predict(current_spend)
        new_roas = predict(new_spend)
        percent_change = ((new_roas - current_roas) / current_roas) * 100
        
        # Classification
        if new_roas >= current_roas:
            classification = "Efficient"
            color = "green"
        elif percent_change >= -caution_threshold:
            classification = "Caution"
            color = "orange"
        else:
            classification = "Not Recommended"
            color = "red"
        
        # Display results
        st.subheader("Results")
        col1, col2, col3 = st.columns(3)
        col1.metric("Current ROAS", f"{current_roas:.2f}")
        col2.metric("Predicted ROAS", f"{new_roas:.2f}", f"{percent_change:.2f}%")
        col3.markdown(f"<div style='background-color:{color};padding:10px;border-radius:5px;color:white;text-align:center;font-weight:bold;'>Recommendation: {classification}</div>", unsafe_allow_html=True)
        
        # Visualization
        st.subheader("ROAS Curve")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot actual data
        ax.scatter(spend, roas, alpha=0.5, label="Historical Data")
        
        # Plot model curve
        x_curve = np.linspace(min(spend) * 0.8, max(spend) * 1.2, 100)
        y_curve = [predict(x) for x in x_curve]
        ax.plot(x_curve, y_curve, 'r-', label=f"{model_type} Model")
        
        # Mark current and proposed points
        ax.scatter([current_spend], [current_roas], color='green', s=100, label=f"Current: ${current_spend:.2f}")
        ax.scatter([new_spend], [new_roas], color='orange', s=100, label=f"Proposed: ${new_spend:.2f}")
        
        ax.set_xlabel("Ad Spend ($)")
        ax.set_ylabel("ROAS")
        ax.set_title("Ad Spend vs ROAS Model")
        ax.grid(alpha=0.3)
        ax.legend()
        
        st.pyplot(fig)
        
        # Model details
        with st.expander("Model Details"):
            st.write(f"**Model Equation**: {equation}")
else:
    st.info("Please upload a CSV file with your ad spend and ROAS data")
    
    # Show sample data
    st.subheader("Sample Data Format")
    sample_data = pd.DataFrame({
        'Date': pd.date_range(start='2023-01-01', periods=10),
        'Spend': [500, 550, 600, 650, 700, 750, 800, 850, 900, 950],
        'ROAS': [4.0, 3.9, 3.8, 3.7, 3.6, 3.5, 3.4, 3.3, 3.2, 3.1]
    })
    st.dataframe(sample_data)
