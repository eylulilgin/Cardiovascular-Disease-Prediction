# Cardiovascular Disease Prediction

## Overview
This project implements a **Neural Network (MLP) model** for predicting **Cardiovascular Disease (CVD)** based on patient health data. The model is trained using **TensorFlow/Keras** and includes data preprocessing steps like scaling, encoding, and imputation.

## Features
- **Preprocesses dataset** (handles missing values, scales numerical features, encodes categorical variables).
- **Splits data into training and test sets**.
- **Implements a Multi-Layer Perceptron (MLP) model** using TensorFlow.
- **Trains the model and evaluates test accuracy**.

## Installation
To set up and run this project, follow these steps:

### **1. Clone the repository**
```bash
git clone https://github.com/yourusername/cvd-prediction.git
cd cvd-prediction
```

### **2. Install Dependencies**
```bash
pip install numpy pandas scikit-learn tensorflow
```

### **3. Run the Script**
```bash
python cvd_prediction.py
```

## Usage
1. Modify `cvd_prediction.py` to use a different dataset if needed.
2. Run the script to train the model.
3. Check test accuracy in the terminal output.

## Example Output
```
Train Accuracy: 87.4%
Test Accuracy: 84.1%
```

## File Renaming Guide
| **Old Name**        | **New Name (Suggested)**  | **Description** |
|---------------------|-------------------------|----------------|
| `cardiovasküler.py` | **`cvd_prediction.py`**  | Main Python script for CVD prediction |

## License
This project is licensed under the **MIT License**.

## Contributions
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`feature-new-feature`).
3. Commit and push your changes.
4. Open a pull request.

## Contact
For any questions or support, please open an issue on GitHub.
