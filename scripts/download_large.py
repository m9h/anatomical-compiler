import sys
import requests
from pathlib import Path
from tqdm import tqdm

def download_file(url, dest):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    # Check existing size for resume
    existing_size = dest.stat().st_size if dest.exists() else 0
    
    headers = {}
    if existing_size > 0:
        headers['Range'] = f'bytes={existing_size}-'
        
    print(f"Downloading {url} to {dest}")
    if existing_size > 0:
        print(f"  Resuming from {existing_size / 1e9:.2f} GB")
        
    response = requests.get(url, headers=headers, stream=True)
    
    if response.status_code == 416: # Range Not Satisfiable
        print("  File already complete or range error.")
        return
        
    total_size = int(response.headers.get('content-length', 0)) + existing_size
    mode = 'ab' if existing_size > 0 and response.status_code == 206 else 'wb'
    
    with open(dest, mode) as f:
        with tqdm(total=total_size, initial=existing_size, unit='B', unit_scale=True, desc=dest.name) as pbar:
            for chunk in response.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python download_large.py <url> <dest>")
        sys.exit(1)
    download_file(sys.argv[1], sys.argv[2])
