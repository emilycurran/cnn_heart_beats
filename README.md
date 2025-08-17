# Final Year Project

## Getting Started

To make it easy for you to get started with this project, follow the steps below.

## Installation

1. Clone the repository:
   ```bash
   git clone https://gitlab.cs.nuim.ie/u210398/final-year-project.git
   cd final-year-project
   ```
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. Ensure mit-bih-arrhythmia-database-1.0.0 is in the same folder as the python scripts.

## Running the Project

This project follows a structured pipeline for ECG classification using 1D and 2D CNN models.

### **1. Generate and Preprocess Data**
Run the following command to process and augment the ECG data:
```bash
python gen_data.py
```
This will:
- Extract ECG signals from the MIT-BIH Arrhythmia Database
- Perform data augmentation on underrepresented arrhythmia classes
- Store the preprocessed data in an SQLite database

### **2. Train the Models**
Train both the **1D CNN** and **2D CNN** models using:
```bash
python train_models.py
```
This script will:
- Split the data into training and validation sets
- Train the models using early stopping
- Save the best-performing models

### **3. Evaluate the Models**
Run the following command to evaluate model performance:
```bash
python evaluate_model.py
```
This will:
- Compute **accuracy, sensitivity, and confusion matrices**
- Generate **ROC and Precision-Recall curves**
- Save evaluation results to files

### **4. Compare Model Performance**
To compare the **inference time, model size, and parameter count** between 1D and 2D CNNs:
```bash
python compare_cnn_cost.py
```

### **5. Run the Entire Pipeline**
To execute all steps (data preprocessing, training, evaluation, and comparison) at once:
```bash
python run_pipeline.py
```

## Project Structure
```
final-year-project/
│── gen_data.py           # Data extraction and augmentation
│── train_models.py       # Train 1D and 2D CNN models
│── evaluate_model.py     # Evaluate model performance
│── compare_cnn_cost.py   # Compare model efficiency
│── run_pipeline.py       # End-to-end execution script
│── paper_model.py        # 2D CNN model definition
│── paper_model_1d.py     # 1D CNN model definition
│── utils.py              # Helper functions
│── README.md             # This file
│── mit-bih-arrhythmia-database-1.0.0 #Extracts data from here, make sure it's in the same directory as gen_data.py
```

## Dependencies
- Python 3.7+
- TensorFlow
- NumPy
- Matplotlib
- SQLite3
- SciPy
- Seaborn
- Scikit-learn




