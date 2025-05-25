import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def print_model_evaluation(model_name, accuracy, sensitivity, specificity, cm):
    print(f"\n{model_name} Model Evaluation:")
    print(f"{model_name} Accuracy: {accuracy:.4f}")
    print(f"{model_name} Sensitivity (Recall): {sensitivity:.4f}")
    print(f"{model_name} Specificity: {specificity:.4f}")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Malignant"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"{model_name} Confusion Matrix")
    plt.show()
