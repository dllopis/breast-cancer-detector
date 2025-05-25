from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
from utils.evaluation import print_model_evaluation

def decision_tree_model(X_train, X_test, y_train, y_test, feature_names):
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    plt.figure(figsize=(20, 10))
    plot_tree(clf, filled=True, feature_names=feature_names, class_names=["Benign", "Malignant"])
    plt.title("Decision Tree for Breast Cancer Diagnosis")
    plt.show()

    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    accuracy = accuracy_score(y_test, y_pred)
    sensitivity = recall_score(y_test, y_pred)
    specificity = tn / (tn + fp)

    print_model_evaluation("Decision Tree", accuracy, sensitivity, specificity, cm)
