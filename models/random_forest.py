from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from scipy.stats import mode
import numpy as np
import matplotlib.pyplot as plt
from utils.evaluation import print_model_evaluation

def random_forest_ensemble_model(X_train, X_test, y_train, y_test, feature_names):
    depths = [3, 5, 7]
    trees = [DecisionTreeClassifier(max_depth=d, random_state=i+1) for i, d in enumerate(depths)]

    preds = []
    for clf, depth in zip(trees, depths):
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        preds.append(pred)

        plt.figure(figsize=(15, 6))
        plot_tree(clf, filled=True, feature_names=feature_names, class_names=["Benign", "Malignant"])
        plt.title(f"Decision Tree with max_depth = {depth}")
        plt.show()

    ensemble_preds = mode(np.vstack(preds), axis=0).mode.flatten()
    cm = confusion_matrix(y_test, ensemble_preds)
    tn, fp, fn, tp = cm.ravel()
    accuracy = accuracy_score(y_test, ensemble_preds)
    sensitivity = recall_score(y_test, ensemble_preds)
    specificity = tn / (tn + fp)

    print_model_evaluation("Ensemble", accuracy, sensitivity, specificity, cm)
