
# 🌾 Nitrogen Fertilizer Recommendation System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Model-orange)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Jupyter Notebook](https://img.shields.io/badge/Notebook-Jupyter-F37626)](https://jupyter.org/)

---

## 🧠 Project Overview
This project builds a **Machine Learning-based Nitrogen Fertilizer Recommendation System** to help farmers and agronomists determine the **optimal amount of nitrogen fertilizer (kg/ha)** for crops based on soil and environmental parameters.

By analyzing data such as **temperature, rainfall, soil pH, and humidity**, the model predicts the required nitrogen levels — promoting **precision agriculture**, **cost efficiency**, and **sustainable farming** practices.

---

## 🚀 Features
✅ Predicts nitrogen fertilizer requirement (kg/ha)  
✅ Built using **Python + Scikit-learn**  
✅ Trained and tested in **Jupyter Notebook**  
✅ Model saved as `.pkl` for easy reuse  
✅ Can be deployed via **Flask / Streamlit** web apps  

---

## 🧰 Tech Stack
| Component | Technology Used |
|------------|-----------------|
| **Programming Language** | Python 3.x |
| **Libraries** | Pandas, NumPy, Scikit-learn |
| **Visualization** | Matplotlib, Seaborn |
| **Model Serialization** | Pickle (`.pkl`) |
| **Environment** | Jupyter Notebook |

---

## 📁 Project Structure
```

Nitrogen_Fertilizer_Recommendation/
│
├── Nitrogen_Fertilizer_Recommendation.ipynb   # Main Notebook (Model training and testing)
├── nitrogen_fertilizer_model.pkl               # Trained Machine Learning model
├── requirements.txt                            # Required dependencies
└── README.md                                   # Project documentation

````

---

## ⚙️ How It Works

1️⃣ **Data Preprocessing**  
Cleans and normalizes agricultural and soil data.  

2️⃣ **Feature Engineering**  
Selects relevant input features such as temperature, rainfall, soil pH, humidity, and crop type.  

3️⃣ **Model Training**  
Trains a regression model (e.g., RandomForestRegressor / Linear Regression) to predict nitrogen requirements.  

4️⃣ **Evaluation**  
Evaluates the model using metrics like R², MAE, and MSE.  

5️⃣ **Deployment**  
The final trained model is saved as `nitrogen_fertilizer_model.pkl` for real-world use.

---

## 💻 Example Usage

### 1. Load the Trained Model
```python
import pickle

with open('nitrogen_fertilizer_model.pkl', 'rb') as file:
    model = pickle.load(file)
````

### 2. Make a Prediction

```python
# Example input: [temperature, rainfall, soil_ph, humidity, crop_type_index]
sample_input = [[28, 200, 6.5, 70, 2]]

predicted_nitrogen = model.predict(sample_input)
print("Recommended Nitrogen (kg/ha):", predicted_nitrogen[0])
```

---

## 📊 Model Performance (Example)

| Metric                        | Value |
| ----------------------------- | ----- |
| **R² Score**                  | 0.89  |
| **Mean Absolute Error (MAE)** | 3.45  |
| **Mean Squared Error (MSE)**  | 14.78 |

> Replace these with your actual notebook results.

---

## 🧩 Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/<your-username>/Nitrogen-Fertilizer-Recommendation.git
cd Nitrogen-Fertilizer-Recommendation
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Open Jupyter Notebook

```bash
jupyter notebook Nitrogen_Fertilizer_Recommendation.ipynb
```

---

## 📦 Requirements

```
numpy
pandas
scikit-learn
matplotlib
seaborn
jupyter
```

---

## 🌍 Future Enhancements

* Integrate with **real-time weather APIs** for dynamic predictions
* Build a **web-based dashboard (Flask/Streamlit)**
* Add **multi-nutrient fertilizer recommendation (N, P, K)**
* Deploy the model as an **API endpoint (FastAPI)**

---

## 💡 Real-World Impact

This model supports:

* **Farmers** – to optimize fertilizer use and cost
* **Agronomists** – to study soil nutrient requirements
* **Policy Makers** – to plan sustainable fertilizer distribution


## 🪪 License

This project is licensed under the PATEL VARSHETSHRIKAR.
Feel free to use, modify, and distribute with attribution.

