# 🩺 Breast Cancer Detection Using Machine Learning

Description:
Breast cancer is the most common cancer amongst women in the world. It accounts for 25% of all cancer cases, and affected over 2.1 Million people in 2015 alone. It starts when cells in the breast begin to grow out of control. These cells usually form tumors that can be seen via X-ray or felt as lumps in the breast area.

The key challenges against it’s detection is how to classify tumors into malignant (cancerous) or benign(non cancerous). We ask you to complete the analysis of classifying these tumors using machine learning (with SVMs) and the Breast Cancer Wisconsin (Diagnostic) Dataset.
This project applies various machine learning models to classify breast cancer tumors as **benign** or **malignant** using the **Breast Cancer Wisconsin Diagnostic Dataset**. It implements visualization, evaluation, and comparison of multiple models including Decision Tree, Random Forest (Ensemble), Support Vector Machine (SVM with RBF kernel), and k-Nearest Neighbors (KNN).

Dataset on kaggle: https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset

---

## 📁 Project Structure

breast_cancer_detection/  
├── main.py # Main script to run the models  
├── breast-cancer.csv # Input dataset  
├── requirements.txt # Project dependencies  
├── models/  
│ ├── decision_tree.py  
│ ├── random_forest.py  
│ ├── svm_rbf.py  
│ └── knn.py  
└── utils/  
├── evaluation.py  
└── io_helpers.py  


---

# 🧠 Models Implemented    
| Model                | Description                                                           | 
|:------------------------|:------------------------------------------------------------------:|
| Decision Tree           |    A simple decision tree classifier with full visualization       |
| Random Forest           |     Manual ensemble of decision trees with majority voting         |
| SVM (RBF Kernel)        |    Support Vector Classifier using an RBF kernel with 2D features  |
| K-Nearest Neighbors     |     KNN with both Euclidean and Manhattan distances                |


# 📊 Metrics Evaluated

- Accuracy  
- Sensitivity (Recall)  
- Specificity  
- Confusion Matrix  
- Decision Boundaries (only for svm-rbf)  

## 🛠 Requirements  

- Python 3.8+  
- pandas  
- scikit-learn  
- matplotlib  
- scipy  
- numpy  

## 📌 Notes  
- SVM visualization is performed on two selected features: radius_mean and texture_mean.  
- Confusion matrices are displayed for each model.  
- Models are trained on a 80/20 train-test split with stratification.  
