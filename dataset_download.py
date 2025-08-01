import os
import zipfile
import urllib.request
import tarfile

def download_and_extract(url, dest_folder, zip_name=None, is_tar=False):
    os.makedirs(dest_folder, exist_ok=True)
    filename = zip_name or url.split("/")[-1]
    zip_path = os.path.join(dest_folder, filename)

    if not os.path.exists(zip_path):
        print(f" Downloading {filename}...")
        urllib.request.urlretrieve(url, zip_path)
        print(" Download complete.")

    print(" Extracting...")
    if is_tar:
        with tarfile.open(zip_path, 'r') as tar_ref:
            tar_ref.extractall(dest_folder)
    else:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dest_folder)
    print(" Extraction complete.")

# ===================== Kvasir-SEG =====================
def setup_kvasir_seg():
    print("\n--- Kvasir-SEG ---")
    url = "https://datasets.simula.no/kvasir-seg/Kvasir-SEG.zip"
    download_and_extract(url, "datasets/Kvasir-SEG")

# ===================== CVC-ClinicDB =====================
def setup_cvc_clinicdb():
    print("\n--- CVC-ClinicDB ---")
    url = "https://github.com/cvblab/PolypSeg/archive/refs/heads/master.zip"
    download_and_extract(url, "datasets/CVC-ClinicDB", zip_name="PolypSeg-master.zip")

    # Move files from PolypSeg-master/CVC-ClinicDB to root dataset folder
    src_dir = "datasets/CVC-ClinicDB/PolypSeg-master/CVC-ClinicDB"
    dst_dir = "datasets/CVC-ClinicDB"
    if os.path.exists(src_dir):
        for item in os.listdir(src_dir):
            os.rename(os.path.join(src_dir, item), os.path.join(dst_dir, item))
        import shutil
        shutil.rmtree("datasets/CVC-ClinicDB/PolypSeg-master")
        print(" CVC-ClinicDB is ready.")

# ===================== BUSI (Manual) =====================
def print_busi_instructions():
    print("\n--- BUSI (Manual Download Required) ---")
    print(" Download BUSI from: https://scholar.cu.edu.eg/?q=afahmy/pages/dataset")
    print("➡ Place the extracted folder at: datasets/BUSI/\n")

# ===================== ISIC 2018 (Manual) =====================
def print_isic_instructions():
    print("\n--- ISIC 2018 (Manual Download Required) ---")
    print("🔗 Register and download 'Task 1: Lesion Segmentation' from:")
    print("   https://challenge.isic-archive.com/data/")
    print("➡ Place the following folders under: datasets/ISIC2018/")
    print("   - ISIC2018_Task1-2_Training_Input/")
    print("   - ISIC2018_Task1_Training_GroundTruth/\n")

# ===================== BraTS 2020 (Manual) =====================
def print_brats_instructions():
    print("\n --- BraTS 2020 (Manual Download Required) ---")
    print(" Register and download from:")
    print("   https://www.med.upenn.edu/cbica/brats2020/data.html")
    print("➡ Extract to: datasets/brats2020/\n")

# ===================== MAIN =====================
if __name__ == "__main__":
    os.makedirs("datasets", exist_ok=True)

    setup_kvasir_seg()
    setup_cvc_clinicdb()
    print_busi_instructions()
    print_isic_instructions()
    print_brats_instructions()

