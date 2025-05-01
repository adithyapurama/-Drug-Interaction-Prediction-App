import streamlit as st
import numpy as np
import pandas as pd
import base64
from keras.models import load_model
from streamlit_option_menu import option_menu

# Define base path
BASE_PATH = r"C:\Users\adith\OneDrive\Desktop\AU\project\project"

# Function to set background image
def set_background(image_path):
    with open(image_path, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode()

    bg_style = f"""
    <style>
        .stApp {{
            background: url("data:image/jpg;base64,{encoded_string}") no-repeat center center fixed;
            background-size: cover;
        }}
        .centered-container {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
        }}
    </style>
    """
    st.markdown(bg_style, unsafe_allow_html=True)

# Load necessary data
@st.cache_data
def load_data():
    df_drug_list = pd.read_csv(f"{BASE_PATH}\\drug_list_processed.csv")
    df_side_effects = pd.read_csv(f"{BASE_PATH}\\drug_side_effects_targets.csv")

    # Rename columns
    df_side_effects.columns = ["Index", "DrugBank_ID", "Drug_Name", "Side_Effects", "Target_Sites"]

    # Strip column names
    df_drug_list.columns = df_drug_list.columns.str.strip()
    df_side_effects.columns = df_side_effects.columns.str.strip()

    # Load NumPy files
    drug_fea = np.load(f"{BASE_PATH}\\drug_fea.npy", allow_pickle=True)
    similarity_matrices = np.load(f"{BASE_PATH}\\similarity_matrices.npy", allow_pickle=True)

    return df_drug_list, df_side_effects, drug_fea, similarity_matrices

# Load model
@st.cache_resource
def load_trained_model():
    return load_model(f"{BASE_PATH}\\final_model.keras")

# Function to determine interaction level color
def get_interaction_color(probability):
    if probability < 0.40:
        return "green", "Low Interaction"
    elif 0.40 <= probability < 0.70:
        return "brown", "Moderate Interaction"  # Changed from orange to brown
    else:
        return "red", "High Interaction"

# Prediction function
def predict_interaction(drug1_name, drug2_name, drug_fea, model, similarity_matrices, df_drug_list, drug_info_dict):
    drug_names = list(df_drug_list["Drug_Name"])

    if drug1_name not in drug_names or drug2_name not in drug_names:
        return {"Error": "One or both drugs not found!"}

    idx1, idx2 = drug_names.index(drug1_name), drug_names.index(drug2_name)

    drug1_features, drug2_features = drug_fea[idx1], drug_fea[idx2]
    similarities = [sim[idx1, idx2] for sim in similarity_matrices]

    expected_input_size = model.input_shape[1]
    combined_features = np.concatenate([drug1_features.flatten(), drug2_features.flatten(), similarities])

    # Adjust feature size
    if combined_features.shape[0] > expected_input_size:
        combined_features = combined_features[:expected_input_size]
    else:
        combined_features = np.pad(combined_features, (0, expected_input_size - combined_features.shape[0]))

    combined_features = combined_features.reshape(1, -1)
    prediction = model.predict(combined_features)

    predicted_label = np.argmax(prediction) if prediction.shape[-1] == 2 else int(prediction[0] > 0.5)
    probability = float(prediction[0][1]) if prediction.shape[-1] == 2 else float(prediction[0])

    # Fetch additional details
    drug1_info = drug_info_dict.get(drug1_name, {"Side_Effects": "N/A", "Target_Sites": "N/A"})
    drug2_info = drug_info_dict.get(drug2_name, {"Side_Effects": "N/A", "Target_Sites": "N/A"})

    return {
        "Drug 1": drug1_name,
        "Drug 2": drug2_name,
        "Interaction": "Yes" if predicted_label == 1 else "No",
        "Probability": probability,
        "Drug 1 Side Effects": drug1_info["Side_Effects"],
        "Drug 2 Side Effects": drug2_info["Side_Effects"],
        "Drug 1 Target Sites": drug1_info["Target_Sites"],
        "Drug 2 Target Sites": drug2_info["Target_Sites"]
    }

# Streamlit App UI
def main():
    # Set background image
    set_background(f"{BASE_PATH}\\bg.jpg")

    # Navigation Menu
    selected = option_menu(
        menu_title=None,
        options=["Home", "Prediction", "About Us"],
        icons=["house", "stethoscope", "info-circle"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal"
    )

    # HOME PAGE
    if selected == "Home":
        st.title("Welcome to the Drug Interaction Predictor! 💊")
        st.markdown("""
            This web application helps you predict potential drug-drug interactions.  
            Simply select two drugs, and our deep learning model will analyze their interaction probability,  
            possible side effects, and target sites.

            ### Why use this tool?
            - **AI-powered predictions** based on deep learning.
            - **Quick and easy drug selection** from a vast database.
            - **Informed decision-making** with probability scores and additional drug details.
        """)

    # PREDICTION PAGE
    elif selected == "Prediction":
        st.markdown('<div class="centered-container">', unsafe_allow_html=True)
        st.title("💊 Drug Interaction Prediction")
        st.markdown("Enter two drugs to predict their interaction, probabilities, side effects, and target sites.")

        # Load data
        df_drug_list, df_side_effects, drug_fea, similarity_matrices = load_data()
        model = load_trained_model()

        # Create dictionary for drug side effects & targets
        drug_info_dict = df_side_effects.set_index("Drug_Name").to_dict(orient="index")

        # Select drugs (Centered)
        col1, col2 = st.columns(2)
        with col1:
            drug1 = st.selectbox("Select Drug 1", df_drug_list["Drug_Name"].tolist(), key="drug1")
        with col2:
            drug2 = st.selectbox("Select Drug 2", df_drug_list["Drug_Name"].tolist(), key="drug2")

        # Prediction Button (Centered)
        if st.button("Predict Interaction"):
            try:
                result1 = predict_interaction(drug1, drug2, drug_fea, model, similarity_matrices, df_drug_list, drug_info_dict)
                result2 = predict_interaction(drug2, drug1, drug_fea, model, similarity_matrices, df_drug_list, drug_info_dict)

                st.subheader("🔍 Prediction Result")

                if "Error" in result1 or "Error" in result2:
                    st.error("One or both drugs not found!")
                else:
                    final_probability = (result1["Probability"] + result2["Probability"]) / 2
                    color, interaction_level = get_interaction_color(final_probability)

                    # Display results with color coding
                    st.markdown(f'<p style="color:{color}; font-size:20px;">'
                                f'**Interaction:** {result1["Interaction"]} '
                                f'({interaction_level}) with final probability: {final_probability:.2%}</p>',
                                unsafe_allow_html=True)

                    st.subheader("⚠ Side Effects")
                    st.write(f"**{drug1}:** {result1['Drug 1 Side Effects']}")
                    st.write(f"**{drug2}:** {result1['Drug 2 Side Effects']}")

                    st.subheader("🎯 Target Sites")
                    st.write(f"**{drug1}:** {result1['Drug 1 Target Sites']}")
                    st.write(f"**{drug2}:** {result1['Drug 2 Target Sites']}")

            except Exception as e:
                st.error(f"Error: {e}")

    # ABOUT US PAGE
    elif selected == "About Us":
        st.title("About Us 🏥")
        st.markdown("""
            This application was developed to assist healthcare professionals, researchers, and patients  
            in identifying potential drug interactions. The system leverages deep learning models trained  
            on extensive datasets to provide accurate predictions.  

            ### Features:
            - **AI-based drug-drug interaction prediction.**
            - **Insightful probability scores for interactions.**
            - **Information on drug side effects and target sites.**
        """)

if __name__ == "__main__":
    main()
