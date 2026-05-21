🚀 Project Overview :

What it does:

  -An end-to-end ML pipeline that classifies financial transactions as legitimate or fraudulent, featuring an interactive Streamlit dashboard for real-time risk assessment and SHAP-based model transparency.

Why it was built:

  -To demonstrate a professional, scalable approach to solving binary classification challenges in imbalanced datasets, spanning the entire lifecycle from data preprocessing to deployment.

Real-world problem it solves:

   -It tackles the "needle in a haystack" problem in fraud detection. By replacing rigid rule-based systems with predictive intelligence, it reduces false positives, streamlines manual audits through risk segmentation, and provides explainable results to improve operational efficiency.

📂 Dataset Information :

   -Due to size constraints, the raw dataset is not included in this repository. The analysis performed in this project relies on a large-scale financial transaction dataset with the following characteristics:

  -Dataset Overview: Contains millions of transaction records representing real-world financial activity.

Key Features : Includes transaction timestamps, merchant categories, customer demographics, and transaction amounts.

Preprocessing : The data was normalized and cleaned to handle class imbalance, which is critical for fraud detection.

Data Access : You can access the sample subset and the processed training data via the link below:

https://www.kaggle.com/c/ieee-fraud-detection/data

🛠️ Key Features : 

   -Predictive Modeling: High-precision LightGBM classification for fraud identification.

   -Risk Segmentation: Categorizes transactions into "Clear," "Suspicious," and "Critical" tiers.

   -Explainable AI: Utilizes SHAP values to provide transparency into the model's decision-making.

Interactive Dashboard : A Streamlit-powered interface for real-time risk assessment and data visualization.

🛠️ Tech Stack :

Frontend : Streamlit (Dashboard UI)

Backend : Python (Logic & API)

Database :CSV/Parquet (Flat-file storage)

Machine Learning : 
  -Python

  -Scikit-learn

  -LightGBM (Gradient Boosting)

  -SHAP (Explainability)

📊 Methodology :

  -The project follows a professional data science lifecycle:

  -EDA: Analyzing transaction distributions and temporal patterns.

  -Feature Engineering: Creating time-based features to improve model sensitivity.

Model Optimization: Fine-tuning classification thresholds to balance Precision vs. Recall.

📈 Model Performance :

| Model    | Accuracy | Precision | Recall | F1 Score |
| -------- | -------- | --------- | ------ | -------- |
| LightGBM | 99.4%    | 97%       | 95%    | 96%      |
| XGBoost  | 99.2%    | 95%       | 94%    | 94%      |

📷 Output Screens :

<img width="1790" height="490" alt="output model comparision 1" src="https://github.com/user-attachments/assets/e47141de-61a0-4e2d-973a-c4dc7503d912" />

<img width="1589" height="590" alt="output model comparision 2" src="https://github.com/user-attachments/assets/f746ab24-d0ad-41e8-869a-c1d65247ab78" />

<img width="766" height="940" alt="SHAP summary" src="https://github.com/user-attachments/assets/b7bc83f3-198e-430e-8b07-845778c760f2" />

📈 Results : 

Best Model :

  -LightGBM was chosen as the final model, achieving the best performance with a 96% F1-Score and outperforming XGBoost in precision and recall.

Key Insights :

   -SHAP analysis showed that transaction frequency, transaction time, and high-risk merchant categories were the major indicators of fraud.

Business Impact :

   -The system reduces false positives and helps security teams identify fraudulent transactions faster, minimizing financial losses.

Final Outcome :

  -A complete fraud detection pipeline was developed with a real-time, explainable Streamlit dashboard for efficient risk analysis and decision-making.

🌟 Future Improvements :

   -Deploy the application on cloud platforms like AWS, Azure, or Google Cloud for better accessibility.

   -Add real-time transaction streaming using Apache Kafka.

   -Improve scalability with Docker and Kubernetes.

  -Integrate a mobile application for instant fraud alerts and user actions.

🤝 Contributors :

Shaik Irshad Begum

Role: Project Lead & Lead Developer 

Responsibilities: End-to-end development, including data preprocessing, feature engineering, LightGBM model training, dashboard deployment, and system documentation.

Contact: https://www.linkedin.com/in/shaikirshad23 | shaik23jr1a04d7@gmail.com 

