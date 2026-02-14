import pandas as pd
import mlflow
import dagshub
import matplotlib.pyplot as plt
import os # Penting untuk mengambil token environment
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# --- SOLUSI LOGIN OTOMATIS GITHUB ACTIONS ---
# Mengatur token agar dagshub.init() tidak meminta otorisasi browser
token = os.getenv("DAGSHUB_TOKEN")
if token:
    os.environ["DAGSHUB_USER_TOKEN"] = token

# 1. Inisialisasi ke DagsHub (Sesuai kriteria Advance untuk simpan online)
dagshub.init(repo_owner='EkaAditPrasetyo', repo_name='Eksperimen_SML_EkaAditPrasetyo', mlflow=True) [cite: 28, 29]

# 2. Load Data
df = pd.read_csv('water_potability_clean.csv')
X = df.drop('Potability', axis=1)
y = df['Potability']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Set Nama Eksperimen
mlflow.set_experiment("Tuning_Manual_Advance")

with mlflow.start_run(run_name="Tuning_Manual_Advance"):
    # --- MANUAL LOGGING PARAMETER (Kriteria Skilled/Advance) ---
    n_estimators = 150
    max_depth = 10
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    
    # Training Model
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    
    # --- MANUAL LOGGING METRIK ---
    acc = accuracy_score(y_test, model.predict(X_test))
    mlflow.log_metric("accuracy", acc)
    
    # 4. LOG ARTEFAK UTAMA (Model)
    mlflow.sklearn.log_model(model, "model_tuned")
    
    # --- LOG ARTEFAK TAMBAHAN (Syarat Advance: Minimal 2 artefak tambahan) --- 
    
    # Artefak 1: Feature Importance Plot
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    plt.figure(figsize=(10,6))
    feat_importances.nlargest(10).plot(kind='barh')
    plt.title("Fitur Paling Berpengaruh")
    plt.savefig("feature_importance.png")
    mlflow.log_artifact("feature_importance.png")
    
    # Artefak 2: Classification Report
    report = classification_report(y_test, model.predict(X_test))
    with open("classification_report.txt", "w") as f:
        f.write(report)
    mlflow.log_artifact("classification_report.txt")
    
    print(f"Berhasil! Model dan 2 artefak tersimpan di DagsHub dengan akurasi: {acc}")