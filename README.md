# employee-performance-predictor
#  Employee Performance Predictor

A Machine Learning project to predict employee performance category 
(High / Average / Low Performer) based on various workplace factors.

---

##  Dataset
- 2000 employee records
- 20 features including Age, Department, Salary, Training Hours, etc.
- Target: Performance Category (High / Average / Low Performer)

---

##  Approach
- Exploratory Data Analysis (EDA)
- Missing value imputation (Median & Mode)
- Label Encoding for categorical variables
- Class imbalance handling using SMOTETomek
- Model: XGBoost Classifier

---

##  Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Imbalanced-learn (SMOTETomek)
- Matplotlib, Seaborn, Plotly

---

##  Results
- Validation F1 Score (Weighted): XX%
- Cross Validation Mean F1: XX%

---

##  Files
| File | Description |
|------|-------------|
| `DS_Hackathon.ipynb` | Main Jupyter Notebook |
| `submission.csv` | Final predictions on test data |

---

##  How to Run
1. Clone this repository
2. Install requirements: `pip install pandas scikit-learn xgboost imbalanced-learn`
3. Open `DS_Hackathon.ipynb` in Jupyter
4. Run all cells
