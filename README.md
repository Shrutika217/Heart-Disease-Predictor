# Heart Disease Predictor

A **user-friendly Streamlit web app** that predicts the risk of heart disease based on health parameters using a trained machine learning model. The app provides **probabilities**, **colored progress bars**, and a **visual chart** for better understanding of your heart health.

---

## Motivation

Heart disease is one of the leading causes of death worldwide. Early detection can save lives. This project aims to provide a **simple, interactive, and accessible web application** that helps users estimate their risk of heart disease based on common medical parameters.

---

## Features

- **Predict Risk:** Determines whether a user has **low or high risk** of heart disease.  
- **Probability Visualization:** Displays probabilities with **green/red progress bars** and a bar chart.  
- **Interactive Interface:** Users can input their parameters using sliders, radio buttons, and drop-downs.  
- **Real-Time Feedback:** Immediate prediction after entering data.  
- **Easy to Use:** No coding knowledge required.  

---

## Input Parameters

The following health metrics are used to predict heart disease risk:

| Feature | Description | Input Type |
|---------|-------------|------------|
| Age | User's age in years | Slider |
| Sex | Male or Female | Radio Button |
| Chest Pain Type (cp) | Type of chest pain (Typical, Atypical, Non-Anginal, Asymptomatic) | Selectbox |
| Resting Blood Pressure (trestbps) | In mm Hg | Number Input |
| Serum Cholesterol (chol) | In mg/dl | Number Input |
| Fasting Blood Sugar > 120 mg/dl (fbs) | Yes/No | Radio Button |
| Resting ECG Results (restecg) | Normal/ST-T Abnormality/Left Ventricular Hypertrophy | Selectbox |
| Max Heart Rate Achieved (thalach) | Beats per minute | Number Input |
| Exercise Induced Angina (exang) | Yes/No | Radio Button |
| ST Depression (oldpeak) | Numeric value | Number Input |
| Slope of Peak Exercise ST Segment (slope) | Upsloping, Flat, Downsloping | Selectbox |
| Number of Major Vessels (ca) | 0–3 | Selectbox |
| Thalassemia (thal) | Normal, Fixed Defect, Reversible Defect | Selectbox |

---

## How It Works

1. **User Input:** Users enter health information through the web interface.  
2. **Data Preprocessing:** Inputs are scaled using a saved `StandardScaler` object (`scaler.pkl`).  
3. **Prediction:** Scaled data is passed to a pre-trained ML model (`model.pkl`).  
4. **Output:** The app displays:
   - Predicted class: Low Risk or High Risk  
   - Probabilities for each class  
   - Colored progress bars (green for low risk, red for high risk)  
   - Probability distribution bar chart  

The ML model was trained on the **UCI Heart Disease Dataset** using **classification algorithms** like Logistic Regression.

---

## Author

**Shrutika Gupta**
