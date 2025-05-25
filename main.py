import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from models.decision_tree import decision_tree_model
from models.random_forest import random_forest_ensemble_model
from models.svm_rbf import svm_rbf_model
from models.k_nearest_neighbor import knn_models

from utils.evaluation import print_model_evaluation
from utils.io_helpers import print_dataframe, save_dataframe_to_txt

def main():
    dataframe = pd.read_csv("data/breast-cancer.csv")
    print_dataframe(dataframe.copy(), 10)

    if dataframe['diagnosis'].dtype == object:
        label_encoder = LabelEncoder()
        dataframe['diagnosis'] = label_encoder.fit_transform(dataframe['diagnosis'])

    save_dataframe_to_txt(dataframe.copy())

    X = dataframe.drop(columns=['diagnosis'])
    y = dataframe['diagnosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\nClass distribution in y_train:")
    print(f"Benign (0): {(y_train == 0).sum()}")
    print(f"Malignant (1): {(y_train == 1).sum()}")

    decision_tree_model(X_train, X_test, y_train, y_test, X.columns)
    random_forest_ensemble_model(X_train, X_test, y_train, y_test, X.columns)
    svm_rbf_model(dataframe.copy())
    knn_models(X_train, X_test, y_train, y_test, ks=[3, 5])

if __name__ == "__main__":
    main()
