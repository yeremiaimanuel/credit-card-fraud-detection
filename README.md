# Credit Card Fraud Detection  

## Author  
Yeremia Imanuel Susanto  

## Project Overview  
This project addresses the problem of **credit card fraud detection**, which is a major concern in the financial sector due to the significant economic losses it causes annually. Fraudulent transactions are rare events, making the dataset highly imbalanced and posing challenges for machine learning models.  

The goal of this project is to build and evaluate a machine learning model that can detect fraudulent transactions with high recall and precision, while minimizing false positives.  

---

## Dataset  
- **Source**: [Kaggle – Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
- **Records**: 284,807 transactions  
- **Fraudulent Transactions**: 492 (~0.17%)  
- **Features**:  
  - 28 anonymized PCA-transformed variables (`V1`–`V28`)  
  - 2 untransformed variables: `Time`, `Amount`  
  - 1 target variable: `Class` (0 = Normal, 1 = Fraud)  

---

## Methodology  
1. **Data Exploration**  
   - Verified no missing values.  
   - Observed highly imbalanced class distribution.  
   - Conducted exploratory data analysis (EDA) on `Amount`, `Time`, and PCA features.  
   - Visualized correlation matrix to understand relationships.  

2. **Preprocessing**  
   - Standardized `Amount` and `Time` using `StandardScaler`.  
   - Retained PCA features as they were already standardized.  

3. **Modeling**  
   - Applied **Random Forest Classifier** with stratified train-test split (70:30).  
   - Random state fixed for reproducibility.  

4. **Evaluation Metrics**  
   - Accuracy, Precision, Recall, F1-score.  
   - Confusion Matrix heatmap for error analysis.  
   - Special emphasis on **Recall** for fraud detection, as false negatives are costlier than false positives.  

---

## Results  
- **Accuracy**: 99.93%  
- **Precision (Fraud)**: High – most predicted frauds were correct.  
- **Recall (Fraud)**: Strong – most fraud cases were detected.  
- **F1-score**: Balanced between precision and recall.  
- **Confusion Matrix**: Very few false negatives, which is critical in fraud detection.  

---

## Conclusion  
The Random Forest Classifier demonstrated excellent performance on the dataset, making it a strong candidate for deployment. However, in real-world applications:  
- Continuous monitoring is required to handle concept drift.  
- Additional models (e.g., XGBoost, Logistic Regression with class weighting) should be tested.  
- Advanced imbalance techniques (SMOTE, anomaly detection) could further improve performance.  
- Cost-sensitive evaluation is essential to minimize financial losses beyond classification error.  

---

## Skills Demonstrated  
- Data preprocessing and cleaning  
- Handling imbalanced data  
- Exploratory Data Analysis (EDA) and visualization  
- Ensemble modeling (Random Forest)  
- Model evaluation and performance analysis  

---

## How to Run  
1. Clone this repository.  
2. Install required dependencies from `requirements.txt`.  
3. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the appropriate directory.  
4. Run the Jupyter Notebook:  
   ```bash
   jupyter notebook credit-card-fraud-detection.ipynb
