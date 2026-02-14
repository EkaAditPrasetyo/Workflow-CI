import mlflow
import os

# Set environment variable agar bisa akses DagsHub
os.environ["MLFLOW_TRACKING_URI"] = os.environ["MLFLOW_TRACKING_URI"]
os.environ["MLFLOW_TRACKING_USERNAME"] = os.environ["MLFLOW_TRACKING_USERNAME"]
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.environ["MLFLOW_TRACKING_PASSWORD"]

# Setup nama Docker Image
docker_user = os.environ["DOCKER_USERNAME"]
docker_image_name = f"{docker_user}/submission-ml-eka:latest"

# 1. Cari Run ID Terakhir dari Eksperimen "Water_Potability_Tuning"
print("Mencari run terakhir...")
# Nama eksperimen ini sudah sesuai dengan yang ada di DagsHub kamu
runs = mlflow.search_runs(experiment_names=["Water_Potability_Tuning"])

if not runs.empty:
    # Ambil run paling atas (paling baru dibuat oleh workflow)
    last_run_id = runs.iloc[0].run_id
    print(f"Ditemukan Run ID: {last_run_id}")
    
    # 2. Lokasi model di dalam run tersebut
    model_uri = f"runs:/{last_run_id}/model_tuned"
    print(f"Sedang memproses build Docker dari: {model_uri}...")
    
    # 3. Jalankan perintah build docker
    build_cmd = f"mlflow models build-docker -m {model_uri} -n {docker_image_name}"
    exit_code = os.system(build_cmd)
    
    if exit_code == 0:
        print("✅ Docker Build Sukses!")
    else:
        raise Exception("❌ Gagal build Docker Image")
else:
    raise Exception("Tidak ada run ditemukan. Pastikan training berjalan dulu.")