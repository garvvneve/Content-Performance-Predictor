# 🚀 Content Performance Predictor

AI/ML-based web application that predicts social media content performance using features like platform, post type, timing, and hashtags, enhanced with real-world business rules for better accuracy.

---

## 📌 Problem Statement

Content creators and marketers often struggle to identify:

- Which content format works best  
- When to post  
- How many hashtags to use  

This project solves that by predicting content performance **before posting**.

---

## 🧠 How It Works

### 1️⃣ Feature Engineering
- Cleaned raw dataset  
- Created features like:
  - Day of week  
  - Time slot (morning, evening, etc.)  
  - Hashtag count  
  - Platform encoding  
  - Post type encoding  

---

### 2️⃣ Machine Learning Model
- Algorithm: **Random Forest Classifier**  
- Handles class imbalance using `class_weight="balanced"`  
- Accuracy evaluated using train-test split  

---

### 3️⃣ Business Intelligence Layer 🔥

Prediction is adjusted using real-world rules:

- Instagram performs better on weekends  
- LinkedIn performs better mid-week  
- Hashtag limits optimized per platform  

---

### 4️⃣ Web Application
- Built using **Flask**  
- Interactive UI for user input  
- Real-time prediction with confidence score  

---

## 🛠️ Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib  
- Flask  
- Joblib  

---

## 📂 Project Structure
├── app.py # Flask app
├── feature_engineering.py # Data preprocessing
├── train_model.py # Model training
├── data/ # Datasets
├── model/ # Saved ML models
├── graphs/ # Visual outputs


---

## ⚙️ How to Run


git clone https://github.com/garvvneve/content-performance-predictor.git
cd content-performance-predictor

pip install -r requirements.txt

# Step 1: Prepare data
python feature_engineering.py

# Step 2: Train model
python train_model.py

# Step 3: Run app
python app.py

📊 Model Insights

Generates:

Confusion Matrix
Feature Importance Graph
Performance Distribution
🎯 Key Features

✔ Predict content success before posting
✔ Platform-specific optimization
✔ Business rule integration with ML
✔ Clean and modern UI
✔ End-to-end ML pipeline

🚀 Future Improvements
Deploy using Streamlit / AWS
Add real-time API data (Google Trends)
Improve model using deep learning
📸 Application Preview
<img width="959" height="539" alt="image" src="https://github.com/user-attachments/assets/33635b02-3772-4f1c-bd2e-8574129989b9" />


👨‍💻 Author

Garv Neve
Computer Engineer | Data & ML Enthusiast
