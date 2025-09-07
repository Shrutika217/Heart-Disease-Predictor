import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Heart Disease Predictor", layout="wide")

# Load model and scaler
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("Heart Disease Prediction App")
st.write("Fill in the details below to check your heart disease risk.")

# Mappings for categorical features
cp_options = {
    0: "0 - Typical Angina",
    1: "1 - Atypical Angina",
    2: "2 - Non-anginal Pain",
    3: "3 - Asymptomatic"
}

restecg_options = {
    0: "0 - Normal",
    1: "1 - ST-T Abnormality",
    2: "2 - Left Ventricular Hypertrophy"
}

slope_options = {
    0: "0 - Upsloping",
    1: "1 - Flat",
    2: "2 - Downsloping"
}

thal_options = {
    3: "3 - Normal",
    6: "6 - Fixed Defect",
    7: "7 - Reversible Defect"
}

# Layout
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 20, 100, 50)
    sex = st.radio("Sex", ("Male", "Female"))
    cp_label = st.selectbox("Chest Pain Type (cp)", list(cp_options.values()))
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)
    fbs = st.radio("Fasting Blood Sugar > 120 mg/dl (fbs)", ["0 - No", "1 - Yes"])

with col2:
    restecg_label = st.selectbox("Resting ECG Results (restecg)", list(restecg_options.values()))
    thalach = st.number_input("Max Heart Rate Achieved (thalach)", 60, 220, 150)
    exang = st.radio("Exercise Induced Angina (exang)", ["0 - No", "1 - Yes"])
    oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 10.0, 1.0, step=0.1)
    slope_label = st.selectbox("Slope of Peak Exercise ST Segment (slope)", list(slope_options.values()))
    ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy (ca)", [0, 1, 2, 3])
    thal_label = st.selectbox("Thalassemia (thal)", list(thal_options.values()))

# Convert selections back to numeric
sex = 1 if sex == "Male" else 0
cp = int(cp_label.split(" ")[0])
fbs = int(fbs.split(" ")[0])
restecg = int(restecg_label.split(" ")[0])
exang = int(exang.split(" ")[0])
slope = int(slope_label.split(" ")[0])
thal = int(thal_label.split(" ")[0])

# Collect features
features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                      thalach, exang, oldpeak, slope, ca, thal]])
features_scaled = scaler.transform(features)

# Prediction
if st.button("🔍 Predict"):
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]

    st.subheader("📊 Prediction Result")
    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("💚 Low Risk of Heart Disease")

    # Show probabilities as text + colored progress bars
    st.markdown("### 🧮 Prediction Probabilities")

    # Low Risk - Green
    st.write(f"**Low Risk:** {probability[0]*100:.2f}%")
    st.markdown(
        f"""
        <div style="background-color: #e0e0e0; border-radius: 5px; width: 100%; height: 20px;">
            <div style="width: {probability[0]*100}%; background-color: green; height: 100%; border-radius: 5px;"></div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # High Risk - Red
    st.write(f"**High Risk:** {probability[1]*100:.2f}%")
    st.markdown(
        f"""
        <div style="background-color: #e0e0e0; border-radius: 5px; width: 100%; height: 20px;">
            <div style="width: {probability[1]*100}%; background-color: red; height: 100%; border-radius: 5px;"></div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Probability bar chart (optional)
    st.markdown("### 📈 Probability Distribution")
    fig, ax = plt.subplots()
    ax.bar(["Low Risk", "High Risk"], probability, color=["green", "red"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    st.pyplot(fig)
