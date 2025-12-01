import os, shutil


def rename_file(dir: str, fname: str, last_extension_to_remove: str):
    fpath = os.path.join(dir, fname)
    if not os.path.exists(fpath):
        print("file does not exist")
        return

    if fname.endswith(last_extension_to_remove):
        try:
            l = len(last_extension_to_remove)
            new_name = fname[:-l]
            nfpath = os.path.join(dir, new_name)
            shutil.move(fpath, nfpath)
            print(f"renamed from {fname} -> {new_name}")
        except Exception as e:
            print(f"Failed to rename: {str(e)}")
    else:
        print("Does not contain .txt extension")

if __name__ == "__main__":
    dir = "headers/CSDMC2010"
    current_extension = '.eml.txt'
    last_extension_to_remove = '.txt'
    print(f"to remove {len(last_extension_to_remove)} characters from the end")

    for dir, _, files in os.walk(dir):
        for file in files:
            if file.endswith(current_extension):
                rename_file(dir, file, last_extension_to_remove)
    

    