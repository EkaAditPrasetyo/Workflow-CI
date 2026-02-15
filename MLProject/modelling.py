import os
import sys
import pandas as pd
import mlflow
import dagshub
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# --- 1. SETUP KONEKSI DAGSHUB (SANGAT PENTING UNTUK CI) ---
# Kita pastikan script tidak error jika token belum terset
token = os.getenv("DAGSHUB_TOKEN")
if not token:
    print("[WARNING] DAGSHUB_TOKEN tidak ditemukan di environment variable.")
    # Script tidak kita matikan, tapi mlflow mungkin akan gagal auth jika repo private.

# Set environment variable agar dagshub.init tidak meminta login browser
os.environ["DAGSHUB_USER_TOKEN"] = token if token else ""

# Inisialisasi DagsHub
# REPO_OWNER dan REPO_NAME harus hardcoded atau diambil dari env jika ingin dinamis
dagshub.init(repo_owner='EkaAditPrasetyo', repo_name='Eksperimen_SML_EkaAditPrasetyo', mlflow=True)

# --- 2. LOAD DATA ---
# Penting: Saat 'mlflow run', working directory adalah root folder MLProject.
# Pastikan file csv ada di sebelah file modelling.py ini.
csv_filename = 'water_potability_clean.csv'

if not os.path.exists(csv_filename):
    print(f"[ERROR] File {csv_filename} tidak ditemukan di folder eksekusi.")
    sys.exit(1) # Keluar dengan error agar pipeline CI berhenti (fail)

df = pd.read_csv(csv_filename)
X = df.drop('Potability', axis=1)
y = df['Potability']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. TRAINING & TRACKING (MANUAL - ADVANCE) ---
# Gunakan nama eksperimen yang spesifik untuk CI agar mudah dibedakan
mlflow.set_experiment("CI_Pipeline_Training")

with mlflow.start_run(run_name="Automated_Train_GitHub_Actions"):
    
    # A. Parameter
    n_estimators = 150
    max_depth = 10
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    
    # B. Training
    print("Sedang melatih model...")
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    
    # C. Metrik
    acc = accuracy_score(y_test, model.predict(X_test))
    mlflow.log_metric("accuracy", acc)
    print(f"Model Accuracy: {acc}")
    
    # D. Artefak Utama (Model)
    mlflow.sklearn.log_model(model, "model_tuned")
    
    # E. Artefak Tambahan 1: Feature Importance
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    plt.figure(figsize=(10,6))
    feat_importances.nlargest(10).plot(kind='barh')
    plt.title("Feature Importance (CI)")
    plt.tight_layout()
    plt.savefig("feature_importance_ci.png")
    mlflow.log_artifact("feature_importance_ci.png")
    
    # F. Artefak Tambahan 2: Classification Report
    report = classification_report(y_test, model.predict(X_test))
    with open("classification_report_ci.txt", "w") as f:
        f.write(report)
    mlflow.log_artifact("classification_report_ci.txt")
    
    print("Selesai! Semua artefak telah dikirim ke DagsHub.")