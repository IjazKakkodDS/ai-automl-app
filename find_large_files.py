import os

# Set the folder to scan and the file size limit
project_dir = "."
size_limit = 100 * 1024 * 1024  # 100 MB in bytes

def bytes_to_mb(size_bytes):
    return round(size_bytes / (1024 * 1024), 2)

print("📦 Scanning for files over 100MB...\n")

for root, dirs, files in os.walk(project_dir):
    for file in files:
        filepath = os.path.join(root, file)
        try:
            size = os.path.getsize(filepath)
            if size > size_limit:
                print(f"⚠️ {filepath} — {bytes_to_mb(size)} MB")
        except OSError:
            continue
