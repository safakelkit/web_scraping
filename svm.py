from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from joblib import dump
from PIL import Image
import os
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

base_dir = "adress"

def load_images_and_labels(base_dir, target_size=(128, 128)):
    X, y = [], []
    class_names = []
    for label in os.listdir(base_dir):
        label_dir = os.path.join(base_dir, label)
        if os.path.isdir(label_dir):
            class_names.append(label)
            for img_file in tqdm(os.listdir(label_dir)):
                if img_file.endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(label_dir, img_file)
                    img = Image.open(img_path).convert("RGB").resize(target_size)
                    X.append(np.array(img))
                    y.append(label)
    return np.array(X), np.array(y), class_names

X, y, class_names = load_images_and_labels(base_dir)
X = X / 255.0  
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

svm_model = SVC(random_state=42)

accuracies_svm = []
conf_matrices_svm = []
precision_scores_svm = []
recall_scores_svm = []
f1_scores_svm = []

kf = KFold(n_splits=10, shuffle=True, random_state=42)

for train_index, test_index in kf.split(X, y_encoded):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y_encoded[train_index], y_encoded[test_index]

    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    svm_model.fit(X_train_flat, y_train)

    predictions = svm_model.predict(X_test_flat)

    acc = accuracy_score(y_test, predictions)
    accuracies_svm.append(acc)

    conf_matrix = confusion_matrix(y_test, predictions)
    conf_matrices_svm.append(conf_matrix)

    precision = precision_score(y_test, predictions, average='macro')
    recall = recall_score(y_test, predictions, average='macro')
    f1 = f1_score(y_test, predictions, average='macro')

    precision_scores_svm.append(precision)
    recall_scores_svm.append(recall)
    f1_scores_svm.append(f1)

mean_accuracy_svm = np.mean(accuracies_svm)
mean_precision_svm = np.mean(precision_scores_svm)
mean_recall_svm = np.mean(recall_scores_svm)
mean_f1_svm = np.mean(f1_scores_svm)

mean_conf_matrix_svm = np.mean(conf_matrices_svm, axis=0)

results_svm = {
    'Accuracy': mean_accuracy_svm,
    'Precision': mean_precision_svm,
    'Recall': mean_recall_svm,
    'F1-Score': mean_f1_svm
}

results_df_svm = pd.DataFrame(results_svm, index=[0])
results_df_svm.to_csv('svm_results.csv', index=False)

np.savetxt("svm_confusion_matrix.csv", mean_conf_matrix_svm, delimiter=",")

print(f"\nSVM Modeli Sonuçları:")
print(f"Accuracy: {mean_accuracy_svm}")
print(f"Precision: {mean_precision_svm}")
print(f"Recall: {mean_recall_svm}")
print(f"F1-Score: {mean_f1_svm}")

print("\nConfusion Matrix (Ortalama):")
print(mean_conf_matrix_svm)

plt.figure(figsize=(8, 6))
sns.heatmap(mean_conf_matrix_svm, annot=True, fmt=".0f", cmap="Blues", cbar=False)
plt.title("Ortalama Confusion Matrix")
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek")
plt.show()

dump(svm_model, 'svm_model.joblib')
print("\nModel svm_model.joblib olarak kaydedildi.")
