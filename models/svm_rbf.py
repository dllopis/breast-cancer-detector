from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt
from utils.evaluation import print_model_evaluation

def svm_rbf_model(dataframe):
    feature_x, feature_y = "radius_mean", "texture_mean"
    X = dataframe[[feature_x, feature_y]].values
    y = dataframe["diagnosis"].values
    X_scaled = StandardScaler().fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = svm.SVC(kernel="rbf", gamma=0.7, C=1.0)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    accuracy = accuracy_score(y_test, y_pred)
    sensitivity = recall_score(y_test, y_pred)
    specificity = tn / (tn + fp)

    print_model_evaluation("SVM (RBF Kernel)", accuracy, sensitivity, specificity, cm)

    fig, ax = plt.subplots(figsize=(6, 5))
    DecisionBoundaryDisplay.from_estimator(model, X_scaled, cmap=plt.cm.coolwarm, ax=ax, alpha=0.8)
    ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors="k")
    ax.set_title("SVC with RBF Kernel - Decision Boundary")
    plt.show()
