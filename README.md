# 📊 Sales Prediction using Machine Learning


## 🏷️ Project Title

**Sales Prediction using  Machine Learning**


## 🎯 Goal

The goal of this project is to **forecast product sales** based on advertising expenditure (TV, Radio, Newspaper). The model helps businesses optimize advertising budgets and maximize sales potential.



## 🎯 Objectives

* Perform **Exploratory Data Analysis (EDA)** to understand dataset features.
* Train a **Machine Learning model** to predict sales.
* Compare **actual vs predicted sales** to measure accuracy.
* Identify which advertising channel impacts sales the most.
* Provide predictions for **new ad spend data**.



## 📂 Project Structure


SalesPrediction/
│── dataset/
│   └── Advertising.csv
│── SalesPrediction.ipynb
│── README.md
│── requirements.txt



## 🛠️ Technologies Used

- **Python 3.x**
- **Pandas, NumPy** → Data processing
- **Matplotlib, Seaborn** → Data visualization
- **Scikit-learn** → Machine Learning (Random Forest, Train-Test Split, Metrics)
- **Jupyter Notebook**



## 📑 Dataset Overview

-**Dataset**: Advertising dataset
-  **Features (X):**
              * TV advertising spend
              * Radio advertising spend
              * Newspaper advertising spend
- **Target (y):**
              * Sales


## 👀 Data Preview

| TV    | Radio | Newspaper | Sales |
| ----- | ----- | --------- | ----- |
| 230.1 | 37.8  | 69.2      | 22.1  |
| 44.5  | 39.3  | 45.1      | 10.4  |
| 17.2  | 45.9  | 69.3      | 9.3   |
| 151.5 | 41.3  | 58.5      | 18.5  |
| 180.8 | 10.8  | 58.4      | 12.9  |



## ⚙️ Installation & Setup

1. Clone this repository

   ```bash
   git clone https://github.com/priyankaejjana24/SalesPrediction.git
   cd SalesPrediction
   ```
2. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebook

   ```bash
   jupyter notebook SalesPrediction.ipynb
   ```


## 📌 Task Overview (Steps Followed)

- Importing the Necessary Libraries.
- Loading the Advertising Dataset.
- Explorartory Data Analysis.
- Describing the Dataset.
- Shape of the Dataset.
- Size of the Dataset.
- List of All Column Names.
- Correlation Heatmap.
- Pairpolt to see Relationships.
- Define Features (X) and target (y).
- Splitting Dataset.
- Model Training.
- Make predictions.
- Evaluate Model Performance.
- Compare Actual Vs Predicted.
- Scatter plot for Actual vs Predicted.
- Predict for New Data.
- Check Model Coefficients.



## 💡 Insights and Outcomes

* **TV advertising** was the most impactful feature on sales.
* **Radio** had moderate effect, while **Newspaper** had minimal influence.
* Model achieved a good **R² score**, showing strong predictive power.
* Cross-validation confirmed model reliability.
* Businesses can optimize ad spend by focusing more on TV and Radio.



## 🚀 Future Improvements

* Use advanced models (XGBoost, Gradient Boosting, Neural Networks).
* Add real-world features (competitor pricing, seasonal trends, discounts).
* Perform hyperparameter tuning for better accuracy.
* Deploy the model as a **Flask/Django web app**.



## 📬 Contact

👩‍💻 **Author:** Priyanka Keerthana Ejjana
📧 **Email:** priyankaejjana@gmail.com
💻 **GitHub:** https://github.com/priyankaejjana24

