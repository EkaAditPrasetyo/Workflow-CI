import mlflow
import os
import sys

# 1. Setup Environment
os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI", "")
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME", "")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD", "")

dagshub_token = os.getenv("DAGSHUB_TOKEN")
if dagshub_token:
    os.environ["DAGSHUB_USER_TOKEN"] = dagshub_token

# Setup nama Docker Image
docker_user = os.getenv("DOCKER_USERNAME", "user")
docker_image_name = f"{docker_user}/submission-ml-eka:latest"

# 2. Cari Run ID Terakhir
# [PERBAIKAN PENTING]: Nama eksperimen HARUS SAMA dengan yang ada di modelling.py
experiment_name = "CI_Pipeline_Training" 

print(f"Mencari run terakhir dari eksperimen: {experiment_name}...")

try:
    # Kita cari run yang statusnya FINISHED saja untuk keamanan
    runs = mlflow.search_runs(experiment_names=[experiment_name], filter_string="status = 'FINISHED'")
    
    if not runs.empty:
        # Ambil run paling atas (terbaru)
        last_run_id = runs.iloc[0].run_id
        print(f"✅ Ditemukan Run ID: {last_run_id}")
        
        # 3. Lokasi model (harus sama dengan nama folder model di modelling.py)
        model_uri = f"runs:/{last_run_id}/model_tuned"
        print(f"Sedang memproses build Docker dari: {model_uri}...")
        
        # 4. Jalankan perintah build docker
        # Catatan: --enable-mlserver opsional, jika error bisa dihapus. 
        # Kita gunakan command standar mlflow docker build
        build_cmd = f"mlflow models build-docker -m {model_uri} -n {docker_image_name}"
        
        print(f"Menjalankan command: {build_cmd}")
        exit_code = os.system(build_cmd)
        
        if exit_code == 0:
            print(f"✅ Docker Build Sukses: {docker_image_name}")
        else:
            print("❌ Gagal build Docker Image.")
            sys.exit(1) # Return error code agar GitHub Actions merah
    else:
        print(f"❌ Tidak ada run ditemukan di eksperimen {experiment_name}.")
        sys.exit(1)
except Exception as e:
    print(f"Terjadi kesalahan: {e}")
    sys.exit(1)