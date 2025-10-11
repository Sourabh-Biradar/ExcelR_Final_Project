ExcelR Final Project : Telecommunication Churn Prediction Model

Ojective : To build a model that can predict the 'Churners' using provided data.

Steps : 
1. Data ingestion : load dataset
2. Data Analysis  : basic data information & stats with visualizations (histograms , boxplots , countplots , pairplot , heatmap)
3. Data Cleaning  : handled outliers (clipping), skewness (log1p transformation) 
4. Feature engineering : created features ratios
5. Preprocessing : Label encoded , Ordinal encoded , Frequency Encoded , Standardized , Robust scaled , Column Transformer
6. Saved preprocessor as .pkl
7. Model building with hyperparameter tuning : used SMOTE , GridSearchCV , RandomizedSearchCV , Pipeline
8. Model evaluation : Metrics used roc_auc_score , F1_score , Precision , Recall & roc_curve
9. Saved best model as .pkl
10. Deployment : using StreamLit app

- cmd : pip install -r requirements.txt
- cmd : streamlit run streamlit_app.py

* StreamLit link : https://excelr-final-project-telecom-churn.streamlit.app
