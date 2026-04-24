#  Intrusion Detection System (IDS)

A Machine Learning-based Intrusion Detection System that classifies web requests as **normal** or **malicious (attack)** using a trained model. This project includes a simple UI for testing inputs and verifying predictions.

---

##  Features

* 🔍 Detects malicious web requests
* 🤖 Uses trained ML model (Random Forest)
* 🌐 Simple web interface using Flask
* ⚡ Real-time prediction
* 📊 Preprocessing with saved encoders and feature columns

---

##  Tech Stack

* Python
* Flask
* Scikit-learn
* Pandas / NumPy
* Jupyter Notebook

---

##  Project Structure

```
├── app.py                  # Main Flask application
├── IDS.ipynb              # Model training notebook
├── rf_model.pkl           # Trained ML model
├── le_path.pkl            # Label encoder
├── feature_columns.pkl    # Feature columns used during training
├── templates/             # HTML templates for UI
├── test_cases.docx        # Sample test cases
├── .gitignore             # Ignored files (datasets, cache, etc.)
```

---

##  Model Details

* Algorithm: Random Forest Classifier
* Training done in: `IDS.ipynb`
* Dataset: CSIC HTTP dataset *(not included in repo)*

---

##  Note

* Dataset is excluded due to size and privacy
* Ensure `.pkl` files are present before running
* This project is for educational purposes

---


##  Author

**Nithish Selvam**

---

