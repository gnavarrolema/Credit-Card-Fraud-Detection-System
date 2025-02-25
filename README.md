# Fraud Detection Project

This project implements a credit card fraud detection system using Apache Spark and MLflow on Databricks Community Edition. The pipeline includes data preparation, feature engineering, model training, batch inference, and monitoring.

## Setup Instructions

1.  **Environment Setup:**
    
    -   Install Apache Spark and PySpark.
    -   Install MLflow and any additional dependencies (you can list these in a `requirements.txt` file).
    -   Optionally, set up a Databricks Community Edition workspace.
2.  **Project Installation:**
    
    -   Clone the repository to your local machine or Databricks workspace.
    -   Ensure that the project directories and file paths (as specified in the configuration classes) are correctly set.
    -   If needed, generate or download the sample data (`credit_card_transactions.csv`) and place it in the `/data` folder.
3.  **Running the Pipeline:**
    
    -   Execute the notebooks or Python scripts in the following order:
        1.  **Data Preparation:** `01_data_preparation.py`
        2.  **Feature Engineering:** `02_feature_engineering.py`
        3.  **Model Training:** `03_model_training.py`
        4.  **Batch Inference:** `04_batch_inference.py`
        5.  **Monitoring:** `05_monitoring.py`
    -   Use MLflow for experiment tracking during model training.

## Project Structure Explanation

The project is organized as follows:

`/FileStore/fraud_detection/
├── data/
│   └── credit_card_transactions.csv    # Raw transaction data
├── notebooks/
│   ├── 01_data_preparation.py          # Data loading, validation, and cleaning
│   ├── 02_feature_engineering.py       # Feature extraction and engineering
│   ├── 03_model_training.py            # Model training and evaluation using MLflow
│   ├── 04_batch_inference.py            # Batch inference for scoring new transactions
│   └── 05_monitoring.py                 # Monitoring of model performance and data quality
└── models/
    └── fraud_detector/                  # Directory to store trained models` 

-   **data/**: Contains the raw CSV file with credit card transactions.
-   **notebooks/**: Contains Python scripts for each stage of the machine learning pipeline.
-   **models/**: Stores the trained fraud detection model.

## Feature Descriptions

The feature engineering step extracts several useful features from the raw data:

-   **Transaction Velocity:**  
    Measures the number of transactions per time window (e.g., 1 hour, 24 hours, 1 week).
    
-   **Amount Velocity:**  
    Aggregates transaction amounts by calculating the sum, average, standard deviation, and maximum value within a time window.
    
-   **Merchant Profiling:**  
    Computes statistics per merchant category, such as:
    
    -   Average transaction amount.
    -   Standard deviation of transaction amounts.
    -   Number of transactions.
    -   Fraud rate for each merchant.
    -   Peak transaction hours.
-   **Time-Based Features:**  
    Extracts time-related attributes including:
    
    -   Hour of the day.
    -   Day of the week.
    -   Month and year.
    -   Indicators for weekends and nighttime transactions.
    -   Time elapsed since the last transaction.

## Model Performance Metrics

The model training process evaluates the fraud detection model using several key performance metrics:

-   **Area Under the ROC Curve (AUC):**  
    Measures the model's ability to distinguish between fraudulent and non-fraudulent transactions.
    
-   **Area Under the Precision-Recall Curve (AUPR):**  
    Focuses on the performance for the minority class (fraud).
    
-   **Accuracy:**  
    Overall correctness of the model's predictions.
    
-   **F1 Score:**  
    Harmonic mean of precision and recall, providing a balance between the two.
    

These metrics are logged during training using MLflow and are used to compare different model configurations.