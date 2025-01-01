import streamlit as st
from sklearn.dummy import DummyClassifier
import shap
import matplotlib.pyplot as plt
import joblib

rename_dict = {
    "Spinal Anaesthesia": "6:_anaesthesia_Spinal",
    "General Anaesthesia": "6:_anaesthesia_General anaesthesia",
    "Laryngeal Mask using a Tube": "27:_laryngeal_mask_BI_Tube",
    "EEG Usage": "6:_anaesthesia_EEG used (is protective)",
    "Target Controlled Infusion (TCI)": "6:_anaesthesia_TCI (Target Controlled Infusion: drugs on level of brain receptors. Algorithm, so should be better than TIVA weight-based.)",
    "Gynaecology/Obstetrics Surgery": "4:_surgical_speciality_Gyn/Obs",
    "Continuous Temperature Monitoring": "6:_anaesthesia_Cont. Temp. Monitoring",
    "Laryngeal Mask using LMA": "27:_laryngeal_mask_BI_LMA",
    "Eyeblinds/Earplugs Offered": "7:_prevention_Eyeblinds/earplugs_offerede",
    "General Surgery": "4:_surgical_speciality_General",
    "Patient Uses Dentures": "21: patient_uses_dentures_BI",
    "Outpatient Setting": "(5:_setting)_Outpatient",
    "Stress Score (NRS)": "15_stress_nrs_BI",
    "Patient Uses Visual Aids": "19:_patient_uses_visual_aids_BI",
}

stress_mean = 3.131818181818182
stress_std = 2.642602953752549


def plot_shap_waterfall(model, X_test, observation_id, max_display=8):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test.iloc[[observation_id]])
    # Use rename_dict to map the feature names to the original names
    # Change keys with values and vice versa
    rename_dict_inv = {v: k for k, v in rename_dict.items()}
    X_cols = [rename_dict_inv[col] for col in X_test.columns]
    feature_names = X_cols

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=X_test.iloc[observation_id].values,
            feature_names=feature_names,
        ),
        max_display=max_display,
        show=False,
    )
    plt.title(f"SHAP Waterfall Plot for Observation {observation_id}")
    plt.tight_layout()
    return plt


# Load model
model = joblib.load("models/model_xgb.pkl")

# Initialize session state variables
if "page" not in st.session_state:
    st.session_state.page = "main"


# Callback functions
def switch_to_results():
    st.session_state.page = "results"


def switch_to_main():
    st.session_state.page = "main"


# Function to map Yes/No inputs into binary values
def binary_mapping(choice):
    return 1 if choice == "Yes" else 0


# Function to get feature values from inputs
def get_feature_values():
    return [
        binary_mapping(st.session_state.spinal),
        binary_mapping(st.session_state.general),
        binary_mapping(st.session_state.lmt),
        binary_mapping(st.session_state.eeg),
        binary_mapping(st.session_state.tci),
        binary_mapping(st.session_state.gyn),
        binary_mapping(st.session_state.temp),
        binary_mapping(st.session_state.lma),
        binary_mapping(st.session_state.eyeblinds),
        binary_mapping(st.session_state.surgery),
        binary_mapping(st.session_state.dentures),
        binary_mapping(st.session_state.outpatient),
        st.session_state.stress,  # Numerical input
        binary_mapping(st.session_state.visual),
    ]


def on_predict_click():
    st.session_state.feature_values = get_feature_values()
    switch_to_results()


# Function for the Input Page
def main_page():
    st.title("Medical Features Input Interface")

    # Group 1: Anaesthesia-Related Features
    st.subheader("Anaesthesia-Related Features")
    col1, col2 = st.columns(2)

    with col1:
        spinal_anaesthesia = st.radio("Spinal Anaesthesia", ["Yes", "No"], key="spinal")
        general_anaesthesia = st.radio(
            "General Anaesthesia", ["Yes", "No"], key="general"
        )
        eeg_usage = st.radio("EEG Usage", ["Yes", "No"], key="eeg")
        tci = st.radio("Target Controlled Infusion (TCI)", ["Yes", "No"], key="tci")

    with col2:
        laryngeal_mask_tube = st.radio(
            "Laryngeal Mask using a Tube", ["Yes", "No"], key="lmt"
        )
        laryngeal_mask_lma = st.radio(
            "Laryngeal Mask using LMA", ["Yes", "No"], key="lma"
        )
        temp_monitoring = st.radio(
            "Continuous Temperature Monitoring", ["Yes", "No"], key="temp"
        )

    # Group 2: Surgery-Related Features
    st.subheader("Surgery-Related Features")
    col3, col4 = st.columns(2)

    with col3:
        gynae_obstetrics_surgery = st.radio(
            "Gynaecology/Obstetrics Surgery", ["Yes", "No"], key="gyn"
        )
        general_surgery = st.radio("General Surgery", ["Yes", "No"], key="surgery")

    with col4:
        outpatient_setting = st.radio(
            "Outpatient Setting", ["Yes", "No"], key="outpatient"
        )

    # Group 3: Patient-Specific Features
    st.subheader("Patient-Specific Features")
    col5, col6 = st.columns(2)

    with col5:
        uses_dentures = st.radio("Patient Uses Dentures", ["Yes", "No"], key="dentures")
        visual_aids = st.radio("Patient Uses Visual Aids", ["Yes", "No"], key="visual")

    with col6:
        eyeblinds_earplugs = st.radio(
            "Eyeblinds/Earplugs Offered", ["Yes", "No"], key="eyeblinds"
        )

    # Group 4: Numerical Scores
    st.subheader("Numerical Inputs")
    stress_score = st.slider("Stress Score (NRS)", 1, 9, value=5, key="stress")

    st.button("Predict", on_click=on_predict_click)


# Function for the Results Page
def results_page():
    st.title("Prediction Results")

    feature_values = st.session_state.feature_values

    # Display input values
    st.markdown("### Input Values")
    feature_names = [
        "Spinal Anaesthesia",
        "General Anaesthesia",
        "Laryngeal Mask using a Tube",
        "EEG Usage",
        "Target Controlled Infusion (TCI)",
        "Gynaecology/Obstetrics Surgery",
        "Continuous Temperature Monitoring",
        "Laryngeal Mask using LMA",
        "Eyeblinds/Earplugs Offered",
        "General Surgery",
        "Patient Uses Dentures",
        "Outpatient Setting",
        "Stress Score (NRS)",
        "Patient Uses Visual Aids",
    ]
    st.write(dict(zip(feature_names, feature_values)))

    # Create a DataFrame for the single observation
    import pandas as pd

    column_names = [rename_dict[feature] for feature in feature_names]
    # Normalize the stress score
    feature_values[feature_names.index("Stress Score (NRS)")] = (
        feature_values[feature_names.index("Stress Score (NRS)")] - stress_mean
    ) / stress_std
    X_test = pd.DataFrame([feature_values], columns=column_names)

    # Dummy model prediction
    prediction = model.predict_proba(X_test)[0, 1]
    elevated_threshold = 0.2
    st.success(f'The patient has {"an elevated" if prediction > elevated_threshold else "a low"} risk of postoperative delirium. Probability: {prediction*100:.1f}%')

    # Generate and display SHAP plot
    st.markdown("### SHAP Values Explanation")
    try:
        fig = plot_shap_waterfall(
            model, X_test, 0
        )  # 0 is the index of our single observation
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Could not generate SHAP plot: {str(e)}")

    st.button("Go Back", on_click=switch_to_main)


# Render the appropriate page
if st.session_state.page == "main":
    main_page()
elif st.session_state.page == "results":
    results_page()
