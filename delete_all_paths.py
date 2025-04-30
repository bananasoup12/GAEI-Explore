import os

def delete_files_in_directories(root_dir, target_dirs):
    for root, dirs, files in os.walk(root_dir):
        if any(dir_name in root.split(os.sep) for dir_name in target_dirs):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")

if __name__ == "__main__":
    delete_files_in_directories(os.getcwd(), {"paths", "combined_paths"})