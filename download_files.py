import os
import wget

def download_file(url, target_path):
    """
    Download a file from a URL to the target path.
    
    Args:
        url (str): URL of the file to download.
        target_path (str): Path where the downloaded file will be saved.
    """
    try:
        wget.download(url, out=target_path)
        print(f"Downloaded: {os.path.basename(target_path)}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")

def ensure_directory_exists(directory):
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory (str): Directory path.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def main():
    
    sam_model_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    groundingdino_url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
    
    weights_dir = "./GroundingDINO/weights"
    ensure_directory_exists(weights_dir)
    
    sam_model_path = os.path.join(weights_dir, "sam_vit_h_4b8939.pth")
    groundingdino_path = os.path.join(weights_dir, "groundingdino_swint_ogc.pth")
    
    download_file(sam_model_url, sam_model_path)
    download_file(groundingdino_url, groundingdino_path)

if __name__ == "__main__":
    main()

