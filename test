import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import io

# Set page configuration
st.set_page_config(
    page_title="Ad Spend ROAS Calculator",
    page_icon="💰",
    layout="wide"
)

# App title and description
st.title("Ad Spend ROAS Calculator")
st.markdown("""
This tool helps you predict how changes in ad spend will affect your Return on Ad Spend (ROAS).
Upload your historical data to build a model, then simulate different spend scenarios.
""")

# Sidebar for data upload and model settings
with st.sidebar:
    st.header("Upload Data")
    st.markdown("Upload a CSV file with at least 'Date', 'Spend', and 'ROAS' columns.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    st.header("Model Settings")
    model_type = st.selectbox(
        "Select Model Type",
        ["Logarithmic", "Polynomial"],
        help="Logarithmic models are better for classic diminishing returns. Polynomial models can capture more complex relationships."
    )
    
    # Only show polynomial degree if polynomial is selected
    poly_degree = 2
    if model_type == "Polynomial":
        poly_degree = st.slider("Polynomial Degree", min_value=2, max_value=3, value=2,
                              help="Higher degrees can fit more complex patterns but may overfit with limited data.")
    
    classification_threshold = st.slider(
        "Caution Threshold (%)",
        min_value=1, max_value=20, value=10,
        help="If new ROAS is below current ROAS by this percentage, it will be classified as 'Caution'."
    )

# Function to fit logarithmic model
def fit_log_model(spend, roas):
    def log_func(x, a, b):
        return a + b * np.log(x)
    
    # Initial guess for parameters
    p0 = [1.0, -0.2]  # Typical diminishing return has negative b
    
    # Fit the model
    try:
        params, _ = optimize.curve_fit(log_func, spend, roas, p0=p0)
        return params
    except:
        st.error("Failed to fit logarithmic model. Try polynomial model or check your data.")
        return None

# Function to fit polynomial model
def fit_poly_model(spend, roas, degree=2):
    try:
        # Use numpy's polyfit
        params = np.polyfit(spend, roas, degree)
        return params
    except:
        st.error("Failed to fit polynomial model. Check your data.")
        return None

# Function to predict ROAS based on model
def predict_roas(spend, model_params, model_type, degree=2):
    if model_type == "Logarithmic":
        a, b = model_params
        return a + b * np.log(spend)
    else:  # Polynomial
        # Convert params to polynomial function
        return np.polyval(model_params, spend)

# Function to classify ROAS change
def classify_change(current_roas, new_roas, threshold=10):
    percent_change = ((new_roas - current_roas) / current_roas) * 100
    
    if new_roas >= current_roas:
        return "Efficient", "green", f"+{percent_change:.2f}%"
    elif percent_change >= -threshold:
        return "Caution", "orange", f"{percent_change:.2f}%"
    else:
        return "Not Recommended", "red", f"{percent_change:.2f}%"

# Function to create the ROAS curve visualization
def plot_roas_curve(spend_data, roas_data, model_params, model_type, current_spend=None, new_spend=None, poly_degree=2):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot actual data points
    ax.scatter(spend_data, roas_data, color='blue', alpha=0.5, label='Historical Data')
    
    # Generate points for the model curve
    min_spend = max(100, 0.5 * min(spend_data))  # Avoid log(0)
    max_spend = 1.5 * max(spend_data)
    curve_x = np.linspace(min_spend, max_spend, 100)
    
    if model_type == "Logarithmic":
        a, b = model_params
        curve_y = a + b * np.log(curve_x)
    else:  # Polynomial
        curve_y = np.polyval(model_params, curve_x)
    
    # Plot the model curve
    ax.plot(curve_x, curve_y, color='purple', linewidth=2, 
            label=f"{model_type} Model")
    
    # Mark current and new spend points if provided
    if current_spend is not None:
        current_roas = predict_roas(current_spend, model_params, model_type, poly_degree)
        ax.scatter(current_spend, current_roas, color='green', s=100, zorder=5, marker='o', 
                  label=f'Current: ${current_spend:,.2f}, ROAS: {current_roas:.2f}')
    
    if new_spend is not None and new_spend > 0:
        new_roas = predict_roas(new_spend, model_params, model_type, poly_degree)
        ax.scatter(new_spend, new_roas, color='red', s=100, zorder=5, marker='*',
                  label=f'Proposed: ${new_spend:,.2f}, ROAS: {new_roas:.2f}')
    
    # Add labels and legend
    ax.set_title('Ad Spend vs. ROAS Model', fontsize=16)
    ax.set_xlabel('Ad Spend ($)', fontsize=14)
    ax.set_ylabel('ROAS', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    # Format x-axis as currency
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
    
    # Add diminishing returns annotation
    if model_type == "Logarithmic" and b < 0:
        ax.annotate('Diminishing Returns Region', 
                   xy=(curve_x[len(curve_x)//2], curve_y[len(curve_y)//2]),
                   xytext=(curve_x[len(curve_x)//2], max(curve_y) * 0.9),
                   arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                   fontsize=12, ha='center')
    
    return fig

# Main app logic
if uploaded_file is not None:
    # Load and process data
    try:
        df = pd.read_csv(uploaded_file)
        
        # Check if required columns exist
        required_cols = ['Spend', 'ROAS']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"Missing required columns: {', '.join(missing_cols)}")
            st.stop()
        
        # Display data preview
        st.subheader("Data Preview")
        st.dataframe(df.head())
        
        # Display basic stats
        st.subheader("Data Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Data Points", len(df))
        with col2:
            st.metric("Average Spend", f"${df['Spend'].mean():.2f}")
        with col3:
            st.metric("Average ROAS", f"{df['ROAS'].mean():.2f}")
        
        # Fit model
        spend_data = df['Spend'].values
        roas_data = df['ROAS'].values
        
        if model_type == "Logarithmic":
            # Ensure no zero spend values for log model
            valid_indices = spend_data > 0
            if not all(valid_indices):
                st.warning(f"Removed {sum(~valid_indices)} data points with zero spend for logarithmic model.")
                spend_data = spend_data[valid_indices]
                roas_data = roas_data[valid_indices]
            
            model_params = fit_log_model(spend_data, roas_data)
        else:  # Polynomial
            model_params = fit_poly_model(spend_data, roas_data, poly_degree)
        
        if model_params is not None:
            # Add simulator
            st.subheader("Ad Spend Simulator")
            st.markdown("Enter your current spend and a proposed new spend amount to simulate the impact on ROAS.")
            
            col1, col2 = st.columns(2)
            with col1:
                current_spend = st.number_input("Current Daily Ad Spend ($)", 
                                              min_value=0.0, 
                                              value=float(spend_data.mean()),
                                              step=100.0,
                                              format="%.2f")
            with col2:
                new_spend = st.number_input("Proposed Daily Ad Spend ($)", 
                                          min_value=0.0, 
                                          value=float(spend_data.mean()) * 1.2,  # Default to 20% increase
                                          step=100.0,
                                          format="%.2f")
            
            # Calculate predictions
            if current_spend > 0:  # Avoid log(0)
                current_roas = predict_roas(current_spend, model_params, model_type, poly_degree)
                
                if new_spend > 0:
                    new_roas = predict_roas(new_spend, model_params, model_type, poly_degree)
                    
                    # Classify the change
                    classification, color, change_text = classify_change(
                        current_roas, new_roas, classification_threshold)
                    
                    # Display results
                    st.subheader("Simulation Results")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Current ROAS", f"{current_roas:.2f}")
                    with col2:
                        st.metric("Predicted ROAS", f"{new_roas:.2f}", delta=change_text)
                    with col3:
                        st.markdown(f"""
                        <div style="
                            background-color: {color};
                            padding: 10px;
                            border-radius: 5px;
                            color: white;
                            text-align: center;
                            font-weight: bold;
                            margin-top: 21px;
                        ">
                            {classification}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Show visualization
                    st.subheader("ROAS Curve Visualization")
                    fig = plot_roas_curve(spend_data, roas_data, model_params, model_type, 
                                         current_spend, new_spend, poly_degree)
                    st.pyplot(fig)
                    
                    # Model details
                    with st.expander("Model Details"):
                        if model_type == "Logarithmic":
                            a, b = model_params
                            st.markdown(f"**Model Equation**: ROAS = {a:.4f} + {b:.4f} × log(Spend)")
                            st.markdown(f"""
                            **Interpretation**:
                            - The base ROAS is {a:.4f}
                            - Each doubling of ad spend changes ROAS by {b*np.log(2):.4f}
                            """)
                        else:  # Polynomial
                            equation = "ROAS = "
                            for i, param in enumerate(reversed(model_params)):
                                if i == 0:
                                    equation += f"{param:.4f}"
                                elif i == 1:
                                    equation += f" + {param:.4f} × Spend"
                                else:
                                    equation += f" + {param:.4f} × Spend^{i}"
                            st.markdown(f"**Model Equation**: {equation}")
                    
                    # Download model data as CSV
                    with st.expander("Export Model Data"):
                        # Generate curve data points
                        curve_x = np.linspace(min(spend_data) * 0.5, max(spend_data) * 1.5, 100)
                        curve_y = [predict_roas(x, model_params, model_type, poly_degree) for x in curve_x]
                        
                        # Create a DataFrame
                        curve_df = pd.DataFrame({
                            'Spend': curve_x,
                            'Predicted_ROAS': curve_y
                        })
                        
                        # Convert to CSV for download
                        csv = curve_df.to_csv(index=False)
                        st.download_button(
                            label="Download Model Data as CSV",
                            data=csv,
                            file_name="roas_model_data.csv",
                            mime="text/csv",
                        )
            else:
                st.warning("Please enter a positive value for current spend.")
    
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
else:
    # Show example when no data is uploaded
    st.info("Please upload your data to get started. The file should contain at least 'Spend' and 'ROAS' columns.")
    
    # Show sample data format
    st.subheader("Sample Data Format:")
    sample_data = pd.DataFrame({
        'Date': pd.date_range(start='2023-01-01', periods=10),
        'Spend': [500, 550, 600, 650, 700, 750, 800, 850, 900, 950],
        'Revenue': [2000, 2145, 2280, 2405, 2520, 2625, 2720, 2805, 2880, 2945],
        'ROAS': [4.0, 3.9, 3.8, 3.7, 3.6, 3.5, 3.4, 3.3, 3.2, 3.1]
    })
    st.dataframe(sample_data)
    
    # Provide a sample CSV for download
    csv = sample_data.to_csv(index=False)
    st.download_button(
        label="Download Sample CSV",
        data=csv,
        file_name="sample_ad_data.csv",
        mime="text/csv",
    )
