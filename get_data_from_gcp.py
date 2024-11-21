"""
1. Read all the patients from the bucket
2. For each patient figure out the Axial scans
3. Once we've figured out the Axial scans, we can start working on the pre-processing
4. Once we've done the pre-processing we can start working on the model
"""

from google.cloud import storage
from pydicom import dcmread
import pandas as pd
import tempfile
from PatientClass import MRUterusDataset


def get_views(bucket_name:str,prefix:str)->list:

    """
        Return a list of all the views for a patient
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    views = []
    for blob in blobs:
        views.append(blob.name.split("/")[-2])
    return list(set(views))

def get_axial_view(bucket_name:str,prefix:str,dataset:MRUterusDataset) -> pd.DataFrame:
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    axialT2 = pd.DataFrame()
    
    for blob in blobs:
        file_name = blob.download_as_bytes()
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(file_name)
            temp_file_path = temp_file.name
        img = dcmread(f"{temp_file_path}")
        if img.SeriesDescription in ["Ax T2","Axial T2"]:
            # add to patient dataset
            dataset.add_dicom_file(img,has_fibroid=None)
        else:
            print("Not Axial T2 view",prefix)
            return False
    print("Axial View",prefix)
    return True

def get_data(bucket_name:str,prefix:str):
    dataset = MRUterusDataset(data_dir = "imgs/")
    for patient in range(1,11):
        views = get_views(bucket_name,f"{prefix}/PA{patient}/ST0")
        if len(views) > 0:
            for view in views:
                axialT2 = get_axial_view(bucket_name,f"{prefix}/PA{patient}/ST0/{view}",dataset)
                if not axialT2:
                    continue  
        else:
            print("No views found for patient",patient)
            break
    


if __name__ == "__main__":
    get_data("nanavati_data","Priyanshi_Nanavati")