from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from utils.evaluation import print_model_evaluation

def knn_models(X_train, X_test, y_train, y_test, ks=[3, 5]):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    for metric in ['euclidean', 'manhattan']:
        for k in ks:
            knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
            knn.fit(X_train_scaled, y_train)
            y_pred = knn.predict(X_test_scaled)

            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            accuracy = accuracy_score(y_test, y_pred)
            sensitivity = recall_score(y_test, y_pred)
            specificity = tn / (tn + fp)

            model_name = f"K-NN (k={k}, metric={metric})"
            print_model_evaluation(model_name, accuracy, sensitivity, specificity, cm)
