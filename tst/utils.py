import numpy as np
import os
import gdown
import zipfile

def download_missing_test_data(verbose = 0):
    """Download missing model files from Google Drive."""
    models_dir = os.path.join(os.getcwd(), 'tst')
    model_files = ['fixtures.zip']  # replace with actual file names
    files_id = ['1GPqcSG3CKuP90Ed36Gv8DLhl1q4VVrVo']  # replace with actual file IDs or URLs
    
    # Create model directory if it doesnot exist
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    for i, file in enumerate(model_files):
        file_path = os.path.join(models_dir, file)
        if not os.path.exists(file_path):
            print(f'{file} not found. Downloading...')
            file_id = files_id[i]  # replace with the actual file ID or URL
            # Use gdown to download the file from Google Drive
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, file_path, quiet=False)
            if verbose:
                print(f'{file} downloaded successfully!')
            # Extract the downloaded zip file
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(models_dir)
            if verbose:
                print(f'{file} extracted successfully!')
            # Delete the zip file after extraction
            os.remove(file_path)
            if verbose:
                print(f'{file} deleted successfully!')



class Helper():
    @staticmethod
    def is_largely_close(cmp,ref,error_margin = 10):
        diff_mask = abs(ref - cmp) > error_margin
        ref_error_perc = (np.count_nonzero(diff_mask)/diff_mask.size)*100
        return ref_error_perc
    
            