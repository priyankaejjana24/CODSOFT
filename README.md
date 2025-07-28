
# 🚢 Titanic Survival Prediction



## 🎯 Project Title
**Titanic Survival Prediction**



## 📌 Objectives

- To explore and understand the Titanic dataset.
- To preprocess and clean the dataset for machine learning.
- To perform feature engineering to improve model performance.
- To visualize trends and relationships in the data.
- To train and evaluate multiple classification models to predict survival.
- To derive insights and patterns from the historical data.



## 📁 Project Structure

```
TitanicSurvival/
├── TitanicSurvival.ipynb        # Main Jupyter notebook
├── README.md                    # Project documentation
├── data/                        # Contains dataset files
└── outputs/                     # Model results, visualizations, etc.
```



## 💻 Technologies Used

- **Language:** Python
- **Development Environment:** Jupyter Notebook
- **Libraries:**
  - `pandas`, `numpy` – Data manipulation
  - `matplotlib`, `seaborn` – Data visualization
  - `scikit-learn` – Machine learning models and evaluation



## 📊 Dataset Overview

- Source: [Kaggle - Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/data)
- The dataset contains information about the passengers aboard the Titanic, including:
  - PassengerId
  - Pclass (ticket class)
  - Name, Sex, Age
  - SibSp (siblings/spouses aboard)
  - Parch (parents/children aboard)
  - Ticket, Fare, Cabin, Embarked
  - Survived (target variable)



## 👁️ Data Preview

| PassengerId | Pclass | Name                   | Sex    | Age | SibSp | Parch | Fare  | Embarked | Survived |
|-------------|--------|------------------------|--------|-----|-------|--------|-------|----------|----------|
| 1           | 3      | Braund, Mr. Owen Harris| male   | 22  | 1     | 0      | 7.25  | S        | 0        |
| 2           | 1      | Cumings, Mrs. John     | female | 38  | 1     | 0      | 71.28 | C        | 1        |
| ...         | ...    | ...                    | ...    | ... | ...   | ...    | ...   | ...      | ...      |




## ⚙️ Installation & Setup

1. Clone this repository or download the notebook.
2. Make sure Python 3.8+ is installed.
3. Install required libraries using pip:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
4. Run the Jupyter Notebook:
   ```bash
   jupyter notebook TitanicSurvival.ipynb
   ```



## 🧾 Tasks Overview

- Data loading and initial inspection
- Missing value treatment
- Feature engineering (Title, Deck, FamilySize, etc.)
- Categorical encoding and scaling
- Exploratory Data Analysis (EDA)
- Model training: Logistic Regression, Decision Tree, Random Forest
- Model evaluation using accuracy, confusion matrix, etc.



## 💡 Insights and Outcomes

- Female passengers had a significantly higher survival rate than males.
- Passengers in higher classes (Pclass 1) were more likely to survive.
- Family size and fare paid showed patterns in survival.
- Engineered features like **Title** and **Deck** improved model accuracy.



## 🔮 Future Improvements

- Hyperparameter tuning using GridSearchCV or RandomizedSearchCV.
- Use of advanced algorithms like XGBoost or LightGBM.
- Deployment of model using Flask or Streamlit.
- Create a web dashboard for predictions and visualization.



## 📬 Contact

**Author:** Priyanka Keerthana Ejjana  
**Email:** [priyankaejjana@gmail.com](mailto:priyankaejjana@gmail.com)  
**GitHub:** [priyankaejjana24](https://github.com/priyankaejjana24)


