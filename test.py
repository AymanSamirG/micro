import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(page_title="ROAS Diminishing Return Calculator", layout="centered")

# --- Title and Description ---
st.title("Ad Spend ROAS Calculator")
st.markdown("""
This micro calculator helps you simulate how ROAS behaves with increasing ad spend.
We model the diminishing returns using a simple logarithmic function.
""")

# --- Define ROAS Prediction Function (logarithmic model) ---
def predict_roas(spend, a=4, b=-0.6):
    return a + b * np.log(spend)

# --- User Input ---
spend = st.slider("Enter your ad spend ($)", min_value=100, max_value=10000, step=100)

# --- Predict ROAS ---
predicted_roas = predict_roas(spend)
st.markdown(f"### Predicted ROAS: **{predicted_roas:.2f}**")

# --- Performance Feedback ---
if predicted_roas >= 2.5:
    st.success("Your spend is within an efficient range.")
elif 1.5 <= predicted_roas < 2.5:
    st.warning("Returns are starting to diminish. Monitor closely.")
else:
    st.error("ROAS is low. Consider reducing spend.")

# --- Generate ROAS Curve ---
spend_range = np.linspace(100, 10000, 200)
roas_values = predict_roas(spend_range)

# --- Plot ---
fig, ax = plt.subplots()
ax.plot(spend_range, roas_values, label='ROAS Curve', linewidth=2)
ax.axvline(spend, color='red', linestyle='--', label=f'Your Spend: ${spend}')
ax.set_title("Diminishing ROAS Over Ad Spend")
ax.set_xlabel("Ad Spend ($)")
ax.set_ylabel("Predicted ROAS")
ax.legend()
ax.grid(True)

st.pyplot(fig)

# --- Footer ---
st.markdown("""
---
Built by [Mova Insights Hub]  
This is a simplified model for educational and directional use. For client-specific simulations, contact us for a custom solution.
""")
