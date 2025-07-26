# Drug Interaction Prediction App

This project is a Streamlit-based web application that predicts drug-drug interactions using a deep learning model. Users can select two drugs from a dropdown and receive an interaction probability, along with side effects and biological target site information for each drug.

## Features

* ✨ **Drug Interaction Prediction**: Select any two drugs and check for potential interactions.
* ⚖ **Interaction Probability Score**: Get a deep learning-generated prediction score.
* ⚡ **Side Effects & Target Sites**: Visual display of side effects and biological targets.
* 💡 **User-Friendly Interface**: Intuitive UI with background styling.

## How It Works

The app loads a trained Keras model that predicts drug-drug interactions. The prediction is based on precomputed similarity matrices including:

* Chemical structure similarity
* ATC code-based similarity
* Side effect-based similarity
* Enzyme and target-based similarity

These matrices are combined using Similarity Network Fusion (SNF) and converted into feature vectors for each drug pair.

## App Architecture

* **Frontend**: Built with [Streamlit](https://streamlit.io), styled using HTML/CSS for enhanced UX.
* **Backend**: Uses a Keras deep learning model for interaction prediction.
* **Data**: Includes drug list, similarity matrices, side effects, and precomputed features.

## Project Files

* `app.py`: Main Streamlit application file
* `final_model.keras`: Trained deep learning model
* `drug_list_processed.csv`: Drug names available for selection
* `drug_side_effects_targets.csv`: Side effects and targets data
* `drug_fea.npy`: Feature matrix for drugs
* `similarity_matrices.npy`: Set of similarity matrices
* `bg.jpg`: Background image for UI styling

## Installation & Setup

### 1. Install Python and Dependencies

```bash
pip install streamlit pandas numpy keras tensorflow streamlit_option_menu jupyter
```

### 2. Run the Jupyter Notebook (Optional)

To inspect data preprocessing and model training:

```bash
jupyter notebook
```

Open `Untitled (1).ipynb` and run all cells.

### 3. Launch the Web App

Navigate to the project directory:

```bash
cd "C:\Users\adith\OneDrive\Desktop\AU\project"
```

Start the app:

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) if not launched automatically.

## App Sections

* **Home**: Welcome and introduction
* **Prediction**: Input drugs and get interaction prediction
* **About Us**: Description of app purpose and creators

## Troubleshooting

* `ModuleNotFoundError`: Install required modules

```bash
pip install streamlit streamlit_option_menu
```

* `Error: One or both drugs not found!`: Ensure CSV files are in the correct `project` directory

## Authors

* App design & ML model integration: Adithya Purama
* Dataset analysis: Based on DrugBank, KEGG, SIDER databases

---

For questions or contributions, feel free to open a pull request or contact the maintainer.
