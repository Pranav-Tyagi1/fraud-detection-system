# Fraud Detection Data Science Project

A complete machine learning system for detecting fraudulent credit card transactions using multiple classification algorithms.

## Features

- **Synthetic Data Generation**: Creates realistic transaction data with fraud patterns
- **Exploratory Data Analysis**: Comprehensive visualizations and statistical analysis
- **Multiple ML Models**: Logistic Regression, Random Forest, and Gradient Boosting
- **Model Comparison**: Automated evaluation and selection of best model
- **Prediction System**: Easy-to-use interface for detecting fraudulent transactions

## Project Structure

```
fraud-detection/
│
├── fraud_detection.py          # Main project file
├── requirements.txt             # Python dependencies
├── fraud_transactions.csv       # Generated dataset (after running)
├── fraud_detection_eda.png      # EDA visualizations (after running)
└── fraud_detection_results.png  # Model results (after running)
```

## Setup Instructions for VS Code

### Step 1: Install Python
- Download Python 3.8+ from [python.org](https://www.python.org/downloads/)
- During installation, check "Add Python to PATH"

### Step 2: Install VS Code
- Download from [code.visualstudio.com](https://code.visualstudio.com/)
- Install the Python extension in VS Code

### Step 3: Create Project Folder
```bash
mkdir fraud-detection
cd fraud-detection
```

### Step 4: Create Virtual Environment
Open terminal in VS Code (Ctrl + ` or View → Terminal):

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 5: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 6: Run the Project
```bash
python fraud_detection.py
```

## What the Project Does

1. **Generates 10,000 synthetic transactions** with 5% fraud rate
2. **Performs EDA** with 6 visualization plots
3. **Trains 3 machine learning models**:
   - Logistic Regression
   - Random Forest
   - Gradient Boosting
4. **Evaluates models** using multiple metrics:
   - Accuracy
   - F1-Score
   - ROC-AUC
   - Precision-Recall curves
5. **Selects best model** automatically
6. **Tests prediction** on a sample transaction

## Output Files

- `fraud_transactions.csv` - 10,000 transaction records
- `fraud_detection_eda.png` - 6-panel exploratory analysis
- `fraud_detection_results.png` - 6-panel model evaluation

## Key Features in the Dataset

- **amount**: Transaction amount in dollars
- **time_hour**: Hour of day (0-23)
- **transaction_frequency**: Number of recent transactions
- **days_since_last**: Days since last transaction
- **merchant_category**: Category code (1-5)
- **is_night**: Binary flag for nighttime transactions
- **high_frequency**: Binary flag for high transaction frequency
- **is_fraud**: Target variable (0=Normal, 1=Fraud)

## Model Performance

Typical results:
- **Accuracy**: 97-99%
- **F1-Score**: 0.85-0.95
- **ROC-AUC**: 0.95-0.99

## Customization

You can modify parameters in `fraud_detection.py`:

```python
# Change dataset size
df = fraud_detector.generate_synthetic_data(n_samples=20000)

# Adjust fraud rate (currently 5%)
n_normal = int(n_samples * 0.95)  # Change 0.95 to desired rate

# Modify model parameters
'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42)
```

## Using the Trained Model

After training, predict on new transactions:

```python
# Format: [amount, time_hour, freq, days_since, merchant, amount_log, is_night, high_freq]
new_transaction = [250.0, 2.0, 18, 0.3, 3, np.log1p(250.0), 1, 1]

prediction, probability = fraud_detector.predict_transaction(new_transaction)
print(f"Prediction: {'FRAUD' if prediction == 1 else 'NORMAL'}")
print(f"Probability: {probability*100:.2f}%")
