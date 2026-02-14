import mlflow
import os

# 1. Setup Environment (Mengambil langsung dari GitHub Secrets)
# Gunakan os.getenv agar tidak error jika salah satu variabel kosong
os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI", "")
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME", "")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD", "")

# Penting: Tambahkan token DagsHub agar mlflow search_runs tidak minta login browser
dagshub_token = os.getenv("DAGSHUB_TOKEN")
if dagshub_token:
    os.environ["DAGSHUB_USER_TOKEN"] = dagshub_token

# Setup nama Docker Image
docker_user = os.getenv("DOCKER_USERNAME", "user")
docker_image_name = f"{docker_user}/submission-ml-eka:latest"

# 2. Cari Run ID Terakhir dari Eksperimen "Tuning_Manual_Advance"
print("Mencari run terakhir dari eksperimen: Tuning_Manual_Advance...")
try:
    runs = mlflow.search_runs(experiment_names=["Tuning_Manual_Advance"])
    
    if not runs.empty:
        # Ambil run paling atas (terbaru)
        last_run_id = runs.iloc[0].run_id
        print(f"✅ Ditemukan Run ID: {last_run_id}")
        
        # 3. Lokasi model (harus sama dengan nama folder model di modelling.py)
        model_uri = f"runs:/{last_run_id}/model_tuned"
        print(f"Sedang memproses build Docker dari: {model_uri}...")
        
        # 4. Jalankan perintah build docker (Kriteria Advance)
        # Perintah ini akan membungkus model menjadi image Docker [cite: 45]
        build_cmd = f"mlflow models build-docker -m {model_uri} -n {docker_image_name} --enable-mlserver"
        exit_code = os.system(build_cmd)
        
        if exit_code == 0:
            print(f"✅ Docker Build Sukses: {docker_image_name}")
        else:
            raise Exception("❌ Gagal build Docker Image. Cek koneksi ke Docker Daemon.")
    else:
        raise Exception("❌ Tidak ada run ditemukan di DagsHub. Pastikan step training sukses!")
except Exception as e:
    print(f"Terjadi kesalahan: {e}")
    exit(1)