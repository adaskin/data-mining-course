
# **ML in the Wild: From Notebook to Production**

*prepared with DeepSeek AI assistant*


---

## **Table of Contents**
1. **Motivation**  
2. **Key Differences: Lab vs. Production**  
3. **Live Demo: Training a Model**  
4. **Live Demo: Building a Minimal Serving API**  
5. **Hands‑on Exercise / Group Work**  
6. **Monitoring and Maintenance**  
7. **The Human Side: Ethics, Fairness, Communication**  
8. **Wrap‑up & Bridge to Week 14**

---

## **1. Motivation – Why Production Matters**

- A top‑ranked Kaggle model ≠ a working production system  
- Example: A fraud detection model that perfect on historical data fails because **fraud patterns change weekly**  
- From “it predicts well” to “it actually solves a problem” – the **Last Mile** of ML is often the hardest  
- **Today’s goal:** Understand what happens *after* you achieve a good cross‑validation score  

**Think about it:**  
What do you need besides a high F1‑score for a model to be useful in a company? Write down three things.

---

## **2. Key Conceptual Differences**

### **2.1 Training ≠ Inference**
- In `.fit()` we have all data at once → **batch**  
- In production, predictions often arrive one by one → **real‑time**  
- Preprocessing must be **identical** to what was applied during training  

> **Remember Week 5?** `StandardScaler` is fitted **only** on the training data, then used to transform test/production data. Same principle!

- **Term:** **Inference** – the process of using a trained model to make predictions on new data.  
  *[Wikipedia: Statistical Inference](https://en.wikipedia.org/wiki/Statistical_inference)*

### **2.2 Reproducibility**
- Can I re‑run the same notebook 6 months later and get exactly the same model?  
- Essential ingredients:  
  - Keep a `requirements.txt` (library versions)  
  - Use a fixed random seed  
  - Document data preprocessing steps  
- **Term:** **Reproducibility** – obtaining consistent results using the same input data, code, and environment.  
  *[The Turing Way – Reproducible Research](https://the-turing-way.netlify.app/reproducible-research/reproducible-research.html)*

### **2.3 Batch vs. Real‑time Serving**
| Batch Prediction | Real‑time API |
|------------------|---------------|
| All inputs processed overnight | Instant response required |
| Latency not critical | Response in < 100 ms |
| Output: a file | Output: JSON/HTTP response |

- Most course exercises assume batch. Real‑world often needs **REST APIs**.

- **Term:** **API (Application Programming Interface)** – a set of rules allowing one piece of software to talk to another. In ML, usually a web API that receives data and returns predictions.  
  *[What is an API?](https://www.redhat.com/en/topics/api/what-are-application-programming-interfaces)*

### **2.4 Train/Serve Skew**
- Feature distributions in production can drift from training data  
- Example: a “price” column that was in TL suddenly becomes USD – no code change, but model breaks  
- This is where **monitoring** (Week 12!) matters

---

## **3. Live Demo – Part 1: Training a Simple Model**

We will use the **Iris dataset** and a **Random Forest classifier** (you know it from Week 8). No new algorithm, just a recap.

### **3.1 Train and Save the Model**
```python
# train_model.py
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load data
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Split (just as we always did)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train a simple model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model to disk
joblib.dump(model, 'iris_model.joblib')
print("Model saved!")
```

**Explanation:**  
- `joblib` is a library for efficiently serialising Python objects (including NumPy arrays and scikit‑learn models). *[Joblib documentation](https://joblib.readthedocs.io/)*  
- The model is now stored in a file. No retraining needed for inference.

---

## **4. Live Demo – Part 2: Serving the Model with Flask**

**Flask** is a lightweight Python web framework. We’ll build a **single `/predict` endpoint** that accepts JSON input and returns a prediction.

### **4.1 Setup**
```bash
pip install flask joblib scikit-learn
```

### **4.2 Minimal Flask Application**
```python
# app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model once when the app starts
model = joblib.load('iris_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()   # read JSON from the request
    # Expect a JSON like: {"features": [5.1, 3.5, 1.4, 0.2]}
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features).tolist()
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
```

### **4.3 Test the API**
Open another terminal and run:
```bash
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

Expected response:
```json
{"prediction": [0]}
```
(0 corresponds to *setosa* in the Iris dataset)

### **4.4 What’s missing? (Production gaps)**
- **Input validation:** no check that `features` is a list of length 4  
- **Error handling:** what if model input is malformed?  
- **Logging:** no record of incoming requests or predictions  
- **Security:** no authentication, HTTPS  

But this is the **core pattern** – and a perfect starting point.

---

## **5. Hands‑on Exercise (25 minutes)**

Work in small groups of 3‑4. Choose **one** of the tasks below. We’ll discuss afterwards.

### **Option A: System Design (No coding required)**
Design a **pipeline diagram** for a *customer churn prediction* system that retrains monthly and serves via a REST API.  
Include and label:  
- Data ingestion  
- Training pipeline (splitting, scaling, model selection)  
- Model serialisation  
- Serving API  
- Monitoring (where to plug in metrics?)  
- Feedback loop (how to collect new labels?)

Draw on paper / tablet and be ready to show 2‑3 groups.

### **Option B: Code Extension (if you have a laptop)**
Take the `app.py` above and add:
1. **Input validation**: check that `features` is a list of 4 numbers; return a 400 error otherwise.  
2. **Basic logging**: write every incoming request and its prediction to a file `logs.txt` (use Python’s `logging` module).  
3. **Confidence score**: also return the predicted probability (use `model.predict_proba`).

---

### **Option A: Churn Prediction System – Pipeline Design (Model Answer)**

**Required Components**

1. **Data Ingestion**  
   - Sources: Customer database (demographics, account age), transaction logs (purchase history, support tickets), external data (market trends).  
   - Data is batch‑exported monthly to a data lake or feature store.  
   - *Label:* **Churn labels** (customer cancelled within 30 days) are attached from historical records.

---

2. **Training Pipeline (Monthly Batch)**  
   - **Data splitting:** time‑based split (train on earlier months, validate on recent month).  
   - **Preprocessing / Scaling:** `StandardScaler` (fit on train, transform train & val).  
   - **Feature engineering:** recency, frequency, monetary (RFM) features; all derived inside a `Pipeline` to avoid train‑serve skew.  
   - **Model selection:** compare Logistic Regression, Random Forest, XGBoost using cross‑validation on training set.  
   - **Evaluation:** select best model based on AUC‑ROC or F1‑score on validation set; check for class imbalance (if needed, supply class weights).

---

3. **Model Serialisation**  
   - The trained model (including preprocessing steps) is saved using `joblib` or MLflow.  
   - A **model registry** stores the new version; the serving system is updated to use the latest model.

---
4. **Serving API**  
   - A REST endpoint (`POST /predict`) receives a single customer’s features.  
   - The API loads the pre‑trained model from the registry, applies identical preprocessing, and returns:
     - `churn_probability`  
     - `prediction` (0/1 based on threshold)  
   - Runs on a lightweight server (Flask / FastAPI) behind a load balancer.

---

5. **Monitoring**  
   - **Data quality:** log mean, min, max, missing percentage of incoming features every hour. Alert if deviation > 3σ from training baseline.  
   - **Prediction drift:** track daily proportion of predicted churners.  
   - **Performance (when labels arrive):** if a customer actually churns later, compare with prediction and compute online precision/recall.  
   - **Latency and error rate:** 4xx/5xx responses, p95 latency < 100 ms.

---

6. **Feedback Loop**  
   - True churn labels are collected after 30 days and stored in the data warehouse.  
   - At each monthly retraining cycle, these new labels are appended to the training set.  
   - The model is re‑evaluated on the most recent month’s data; if performance drops significantly, an alert is triggered and the model is refreshed.

---

**Example Mermaid Diagram**

```mermaid
flowchart TD
    subgraph Data [Monthly Batch]
        A[(Customer DB)] --> B[Feature Store]
        C[(Transactions)] --> B
    end

    subgraph Training [Training Pipeline (Monthly)]
        B --> D[Time-based Split]
        D --> E[Preprocessing & Scaling]
        E --> F[Model Selection (CV)]
        F --> G[Evaluation & Validation]
        G --> H[Model Registry (joblib)]
    end

    subgraph Serving [Real-time API]
        I[Client App] -->|POST /predict| J[Flask API]
        J <-->|Load model| H
        J --> K[Response: churn prob, prediction]
    end

    subgraph Monitoring [Monitoring & Drift]
        J --> L[Log features & predictions]
        L --> M[(Monitoring DB)]
        M --> N[Drift Detection & Alerts]
        O[Actual Churn Labels] -->|Feedback after 30d| M
    end

    H -->|Model update| J
    O -->|New training data| B
```

---

## **6. Monitoring and Maintenance**

**Monitoring** answers: *Is my live model still working as expected?*

### **6.1 What to Monitor (No New Tools)**
- **Feature drift**: compute mean and variance of incoming features every hour. If the mean of “sepal length” jumps from 5.8 to 8.0 → alarm!  
  (You already know how to do this – `numpy.mean`, `numpy.std`.)
- **Prediction distribution**: if a classifier suddenly always predicts class 0, something is broken (recall Week 5 evaluation metrics).  
- **Performance metrics over time**: log true labels (if available) and compute accuracy, precision, recall daily.

**Connect to Week 12:**  
Which technique can automatically flag abnormal incoming samples?  
**Anomaly detection** – exactly what we covered in clustering & anomaly detection week. You can use a simple threshold on reconstruction error or density.

### **6.2 Concept Drift**
- **Definition:** The statistical properties of the target variable, which the model is trying to predict, change over time in unforeseen ways.  
  *[Wikipedia: Concept drift](https://en.wikipedia.org/wiki/Concept_drift)*  
- **Example:** A movie recommender trained before a pandemic faces entirely new user habits.  
- **Mitigation:** Regular retraining, model re‑evaluation, and monitoring.

---

## **7. The Human Side: Ethics, Fairness, Communication**

### **7.1 Model Cards**
- A **model card** is a short document that describes:
  - Intended use  
  - Training data (and its limitations)  
  - Evaluation results (including fairness metrics)  
  - Ethical considerations  
  *[Model Cards for Model Reporting (Google)](https://modelcards.withgoogle.com/about)*

**Exercise:** For the Iris model we deployed, write a 2‑sentence model card. Who should use it? What are the risks?

### **7.2 Communicating with Stakeholders**
- You must explain to software engineers, product managers, and possibly regulators:
  - What the model does (in simple words)
  - Its **limitations** (e.g., only trained on flowers from 1936)
  - How to detect when it fails
- Use metrics from **Week 5** – accuracy might not be enough, contextualise them.

---

## **8. Wrap‑up**

**Key Takeaways:**
- A `.joblib` file is just the beginning; serving, monitoring, and maintenance turn a model into a product.
- You already have all the conceptual tools: scaling, splitting, evaluation, anomaly detection.
- Production ML is **not** about new algorithms, it’s about **engineering and responsibility**.

---

**For your project presentations:**
- Include **one slide** (optional but recommended) on:
  - “If this were a real system, how would you serve it? How would you know it’s still working next month? Who would use it, and what are the risks?”

---

**Suggested Reading (if curious):**
- [Rules of Machine Learning: Best Practices for ML Engineering (Google)](https://developers.google.com/machine-learning/guides/rules-of-ml)
- [Hidden Technical Debt in Machine Learning Systems](https://papers.nips.cc/paper_files/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html) (paper reference)
- [Flask quickstart](https://flask.palletsprojects.com/en/stable/quickstart/)

---

**Thank you – and see you next week for the final presentations!**
