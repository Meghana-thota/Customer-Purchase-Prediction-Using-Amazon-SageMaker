# Customer-Purchase-Prediction-Using-Amazon-SageMaker

📌 Project Description
This project builds a machine learning model to predict customer purchase behavior based on historical data. Using Amazon SageMaker, we train and deploy an XGBoost classification model to classify whether a customer is likely to make a purchase (1) or not (0). The project follows a complete MLOps pipeline, including data preprocessing, training, evaluation, and deployment.

Features & Workflow

1️⃣ Data Processing & Preprocessing
Dataset: Customer transaction history with features like age, income, browsing behavior, and previous purchases.
Preprocessing Steps:
Handle missing values.
Normalize and encode categorical features.
Split dataset into train (70%) and test (30%).
Store processed data in Amazon S3 for SageMaker training.

2️⃣ Model Training Using SageMaker
We use Amazon SageMaker's managed XGBoost algorithm for training.
Hyperparameters tuning is done to optimize the model.
Training data is retrieved from S3, and the trained model is saved back to S3.

3️⃣ Model Evaluation & Performance Metrics
Predictions are evaluated using:
Confusion Matrix
Accuracy Score
Precision, Recall, and F1-Score
Expected Accuracy: ~89% based on validation results.

4️⃣ Model Deployment & Real-time Inference
The trained model is deployed using SageMaker endpoints.
Input customer data is serialized in CSV format for prediction.
The model predicts whether a customer is likely to purchase (1) or not (0).

5️⃣ Results & Insights
Overall Classification Rate: ~89.7%
Purchase Prediction Accuracy: 66%
False Positive Rate: 34% (Optimized via hyperparameter tuning)

Technologies & Tools Used
✅ Amazon SageMaker – Model training & deployment
✅ AWS S3 – Cloud storage for datasets & models
✅ XGBoost – ML algorithm for classification
✅ Pandas & NumPy – Data preprocessing & manipulation
✅ Matplotlib & Seaborn – Data visualization
✅ Boto3 – AWS SDK for interacting with S3 & SageMaker
✅ Scikit-learn – Model evaluation & metrics


