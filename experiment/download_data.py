import os, requests, zipfile, io, shutil, sys

data_dir = os.path.expanduser('~/.keras/datasets/plantvillage_corn')
os.makedirs(data_dir, exist_ok=True)

url = 'https://github.com/spMohanty/PlantVillage-Dataset/archive/refs/heads/master.zip'
print(f'Downloading PlantVillage from GitHub...', flush=True)
resp = requests.get(url, stream=True, timeout=600)
print(f'Status: {resp.status_code}', flush=True)

if resp.status_code != 200:
    print(f'Failed: {resp.status_code}')
    sys.exit(1)

content = resp.content
print(f'Downloaded: {len(content)//1024//1024} MB', flush=True)

with zipfile.ZipFile(io.BytesIO(content)) as zf:
    corn_files = [n for n in zf.namelist() if 'Corn' in n and 'color' in n
                   and n.lower().endswith(('.jpg','.jpeg','.png'))]
    print(f'Found {len(corn_files)} corn images', flush=True)
    
    count = 0
    for f in corn_files:
        # Extract class from the ZIP path (e.g., ".../Corn_(maize)___Common_rust/xxx.JPG")
        path_parts = f.split('/')
        class_from_path = None
        for part in path_parts:
            if '___' in part:
                class_from_path = part.split('___')[1].replace('(', '').replace(')', '').replace(' ', '_')
                break
        if class_from_path is None:
            continue
        basename = os.path.basename(f)
        cls_dir = os.path.join(data_dir, class_from_path)
        os.makedirs(cls_dir, exist_ok=True)
        dst = os.path.join(cls_dir, basename)
        if not os.path.exists(dst):
            data = zf.read(f)
            with open(dst, 'wb') as fout:
                fout.write(data)
            count += 1
    print(f'Extracted {count} images', flush=True)

print('\nDataset structure:', flush=True)
for d in sorted(os.listdir(data_dir)):
    p = os.path.join(data_dir, d)
    if os.path.isdir(p):
        print(f'  {d}: {len(os.listdir(p))} images', flush=True)
print('Done!', flush=True)
