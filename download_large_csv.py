import requests
from pathlib import Path
from tqdm import tqdm
import os

# NYC Yellow Taxi 2015-01 CSV (about 5GB zipped)
url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2015-01.csv"
output_dir = Path("large_datasets")
output_dir.mkdir(exist_ok=True)
output_file = output_dir / "yellow_tripdata_2015-01.csv"

print(f"Downloading: {url}")
print(f"Saving to: {output_file}")

response = requests.get(url, stream=True)
total_size = int(response.headers.get('content-length', 0))
block_size = 1024 * 1024  # 1MB

with open(output_file, 'wb') as f, tqdm(
    total=total_size, unit='iB', unit_scale=True, desc=output_file.name
) as bar:
    for data in response.iter_content(block_size):
        f.write(data)
        bar.update(len(data))

print("\n✅ Download complete!")
print(f"File size: {os.path.getsize(output_file) / (1024*1024*1024):.2f} GB") 