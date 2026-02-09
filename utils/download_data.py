import os
import sys
import requests
import zipfile
import tarfile
from tqdm import tqdm

KVASIR_V2_URL = "https://datasets.simula.no/downloads/kvasir/kvasir-v2.zip" # Official Link seems to be this structure
# Alternative backup links might be needed if Simula server is slow/down.

def download_file(url, target_path):
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'}
    response = requests.get(url, stream=True, verify=False, headers=headers)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024 # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    
    with open(target_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")
        return False
    return True

def extract_file(archive_path, extract_to):
    print(f"Extracting {archive_path}...")
    if archive_path.endswith("tar.gz"):
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=extract_to)
    elif archive_path.endswith("zip"):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    print("Extraction complete.")

def main():
    data_dir = os.path.join(os.path.dirname(__file__), "../data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    dataset_name = "kvasir-dataset-v2"
    dataset_path = os.path.join(data_dir, dataset_name)
    archive_path = os.path.join(data_dir, f"{dataset_name}.zip")

    if os.path.exists(dataset_path):
        print(f"Dataset already exists at {dataset_path}")
        return

    print(f"Downloading {dataset_name} from {KVASIR_V2_URL}...")
    try:
        if not os.path.exists(archive_path):
            success = download_file(KVASIR_V2_URL, archive_path)
            if not success:
                print("Download failed.")
                return
        
        extract_file(archive_path, data_dir)
        print(f"Setup complete. Data located at {dataset_path}")
        
    except Exception as e:
        print(f"Error during setup: {e}")
        print("Please manually download Kvasir v2 and place it in the data folder.")

if __name__ == "__main__":
    main()
