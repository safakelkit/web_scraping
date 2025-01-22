from sklearn.calibration import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
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

knn_model = KNeighborsClassifier()

accuracies_knn = []
conf_matrices_knn = []
precision_scores_knn = []
recall_scores_knn = []
f1_scores_knn = []

kf = KFold(n_splits=10, shuffle=True, random_state=42)

for train_index, test_index in kf.split(X, y_encoded):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y_encoded[train_index], y_encoded[test_index]
    
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    knn_model.fit(X_train_flat, y_train)
    
    predictions = knn_model.predict(X_test_flat)
    
    acc = accuracy_score(y_test, predictions)
    accuracies_knn.append(acc)

    conf_matrix = confusion_matrix(y_test, predictions)
    conf_matrices_knn.append(conf_matrix)
    
    precision = precision_score(y_test, predictions, average='macro')
    recall = recall_score(y_test, predictions, average='macro')
    f1 = f1_score(y_test, predictions, average='macro')

    precision_scores_knn.append(precision)
    recall_scores_knn.append(recall)
    f1_scores_knn.append(f1)

mean_accuracy_knn = np.mean(accuracies_knn)
mean_precision_knn = np.mean(precision_scores_knn)
mean_recall_knn = np.mean(recall_scores_knn)
mean_f1_knn = np.mean(f1_scores_knn)

mean_conf_matrix_knn = np.mean(conf_matrices_knn, axis=0)

results_knn = {
    'Accuracy': mean_accuracy_knn,
    'Precision': mean_precision_knn,
    'Recall': mean_recall_knn,
    'F1-Score': mean_f1_knn
}

results_df_knn = pd.DataFrame(results_knn, index=[0])
results_df_knn.to_csv('knn_results.csv', index=False)

np.savetxt("knn_confusion_matrix.csv", mean_conf_matrix_knn, delimiter=",")

print(f"\nkNN Modeli Sonuçları:")
print(f"Accuracy: {mean_accuracy_knn}")
print(f"Precision: {mean_precision_knn}")
print(f"Recall: {mean_recall_knn}")
print(f"F1-Score: {mean_f1_knn}")

print("\nConfusion Matrix (Ortalama):")
print(mean_conf_matrix_knn)

plt.figure(figsize=(8, 6))
sns.heatmap(mean_conf_matrix_knn, annot=True, fmt=".0f", cmap="Blues", cbar=False)
plt.title("Ortalama Confusion Matrix")
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek")
plt.show()

dump(knn_model, 'knn_model.joblib')
print("\nModel knn_model.joblib olarak kaydedildi.")

