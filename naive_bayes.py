from sklearn.naive_bayes import GaussianNB
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

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

nb_model = GaussianNB()

accuracies_nb = []
conf_matrices_nb = []
precision_scores_nb = []
recall_scores_nb = []
f1_scores_nb = []

kf = KFold(n_splits=10, shuffle=True, random_state=42)

for train_index, test_index in kf.split(X, y_encoded):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y_encoded[train_index], y_encoded[test_index]
    
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    nb_model.fit(X_train_flat, y_train)

    predictions = nb_model.predict(X_test_flat)

    acc = accuracy_score(y_test, predictions)
    accuracies_nb.append(acc)

    conf_matrix = confusion_matrix(y_test, predictions)
    conf_matrices_nb.append(conf_matrix)
    
    precision = precision_score(y_test, predictions, average='macro')
    recall = recall_score(y_test, predictions, average='macro')
    f1 = f1_score(y_test, predictions, average='macro')

    precision_scores_nb.append(precision)
    recall_scores_nb.append(recall)
    f1_scores_nb.append(f1)

mean_accuracy_nb = np.mean(accuracies_nb)
mean_precision_nb = np.mean(precision_scores_nb)
mean_recall_nb = np.mean(recall_scores_nb)
mean_f1_nb = np.mean(f1_scores_nb)

mean_conf_matrix_nb = np.mean(conf_matrices_nb, axis=0)

results_nb = {
    'Accuracy': mean_accuracy_nb,
    'Precision': mean_precision_nb,
    'Recall': mean_recall_nb,
    'F1-Score': mean_f1_nb
}

results_df_nb = pd.DataFrame(results_nb, index=[0])
results_df_nb.to_csv('nb_results.csv', index=False)

np.savetxt("nb_confusion_matrix.csv", mean_conf_matrix_nb, delimiter=",")

print(f"\nNaive Bayes Modeli Sonuçları:")
print(f"Accuracy: {mean_accuracy_nb}")
print(f"Precision: {mean_precision_nb}")
print(f"Recall: {mean_recall_nb}")
print(f"F1-Score: {mean_f1_nb}")

print("\nConfusion Matrix (Ortalama):")
print(mean_conf_matrix_nb)

plt.figure(figsize=(8, 6))
sns.heatmap(mean_conf_matrix_nb, annot=True, fmt=".0f", cmap="Blues", cbar=False)
plt.title("Ortalama Confusion Matrix")
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek")
plt.show()

dump(nb_model, 'naive_bayes_model.joblib')
print("\nModel naive_bayes_model.joblib olarak kaydedildi.")
