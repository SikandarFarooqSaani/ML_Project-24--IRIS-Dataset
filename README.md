# 🌸 Iris Flower Classification with Voting Classifier  

![Iris Logo](https://img.icons8.com/emoji/96/cherry-blossom.png)  

---

## 📌 Project Overview  
This project is all about predicting the type of **Iris flowers** 🌺 using **Machine Learning models**.  
We take the famous **Iris Dataset** from Kaggle and put multiple classifiers to the test.  
Finally, we combine them using a **Voting Classifier** to see how teamwork (models working together 🤝) can sometimes outperform individual models.  

---

## 📂 Dataset  
📊 **Dataset link:** [Iris Dataset on Kaggle](https://www.kaggle.com/datasets/saurabh00007/iriscsv)  

- No null or duplicate values ✅  
- Removed unnecessary columns like **id** 🗑️  
- Encoded the target labels using **LabelEncoder** 🎯  

---

## ⚙️ Tech Stack & Libraries  
- **Python** 🐍  
- **OS, NumPy, Pandas** → for handling data  
- **Seaborn & Matplotlib** → for beautiful visualizations 📊  
- **Scikit-learn** → ML models and evaluation  

---

## 🖼️ Data Visualization  
We created a **pairplot** for all features to see how the classes separate visually:  

<img width="1059" height="986" alt="24-1" src="https://github.com/user-attachments/assets/72350cf2-1ba8-4a4c-95a1-07c18786eb89" />
 

---

## 🚀 Project Workflow  

1. **Preprocessing**  
   - Dropped unnecessary columns (like `id`)  
   - Encoded target labels using **LabelEncoder**  

2. **Feature Selection**  
   - Extracted only **2 feature columns** to make prediction harder (more challenging task 🎯).  

3. **Train-Test Split**  
   - Divided data into `X` (features) and `y` (target)  
   - Applied **train_test_split**  

4. **Base Models Used**  
   - Logistic Regression  
   - Random Forest  
   - K-Nearest Neighbors (KNN)  

5. **Individual Model Performance (Cross-Val Score)**  
   - Logistic Regression → **0.75**  
   - Random Forest → **0.62**  
   - KNN → **0.61**  

6. **Voting Classifier (Logistic + RF + KNN)**  
   - Hard Voting → **0.67**  
   - Soft Voting → **0.67**  
   - Tried **nested loops for weights** → Best score was **0.70** with weights:  
     - Logistic Regression = 3  
     - Random Forest = 1  
     - KNN = 2  

7. **Support Vector Machines (SVMs)**  
   - Trained **5 different SVMs** with polynomial degrees from 1 → 5  
   - Results:  
     - Degree 1 → 85%  
     - Degree 2 → 85%  
     - Degree 3 → 89%  
     - Degree 4 → 81%  
     - Degree 5 → 86%  

8. **Voting on SVMs**  
   - Combined all SVMs in a **Voting Classifier**  
   - Achieved an impressive **0.93 Cross-Val Score** 🎉  

---

## 📊 Results  

| Model                  | Accuracy |
|-------------------------|----------|
| Logistic Regression     | 0.75     |
| Random Forest           | 0.62     |
| KNN                     | 0.61     |
| Voting (LR + RF + KNN)  | 0.67     |
| Weighted Voting (3-1-2) | 0.70     |
| Best SVM (deg=3)        | 0.89     |
| Voting (All SVMs)       | **0.93** ✅ |

---

## 💡 Key Takeaways  
- Ensemble models can sometimes improve accuracy, but not always 🚦  
- SVM with degree 3 performed the best among single models 🏆  
- Voting multiple SVMs together gave the **highest accuracy (0.93)** 🎯  

---

### 📌 Future Improvements

Try Gradient Boosting / XGBoost for even better performance 🌟

Add hyperparameter tuning with GridSearchCV / RandomizedSearchCV 🔧

Use all 4 features instead of just 2 for full dataset exploration 🌸

### 🙌 Acknowledgements

Kaggle Dataset

Scikit-learn Team for easy ML model implementation

Seaborn & Matplotlib for beautiful visualizations
