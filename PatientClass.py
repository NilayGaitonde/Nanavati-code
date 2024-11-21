import pydicom
from torch import nn
from torch.utils.data import DataLoader,Dataset
import h5py
from dataclasses import dataclass
import pandas as pd
import numpy as np
from pathlib import Path
import torch
from typing import Dict, List, Optional

@dataclass
class PatientInfo:
    """Patient-level information"""
    patient_id: str
    name: str
    age: str
    sex: str
    
@dataclass
class StudyInfo:
    """Study-level information"""
    study_instance_uid: str
    study_date: str
    study_description: str
    
@dataclass
class SeriesInfo:
    """Series-level information"""
    series_instance_uid: str
    series_description: str
    series_number: int
    slice_thickness: float
    pixel_spacing: tuple

class MRUterusDataset(Dataset):
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize different levels of metadata
        self.patient_df = pd.DataFrame(columns=['patient_id', 'name', 'age', 'sex'])
        self.study_df = pd.DataFrame(columns=['study_instance_uid', 'patient_id', 'study_date', 'study_description'])
        self.series_df = pd.DataFrame(columns=['series_instance_uid', 'study_instance_uid', 'series_description', 
                                             'series_number', 'slice_thickness', 'pixel_spacing'])
        self.slice_df = pd.DataFrame(columns=['slice_id', 'series_instance_uid', 'instance_number', 
                                            'slice_location', 'has_fibroid'])
        
        # H5 file for pixel data
        self.h5_path = self.data_dir / 'image_data.h5'
        if not self.h5_path.exists():
            with h5py.File(self.h5_path, 'w') as f:
                pass

    def add_patient_data(self, dicom_slice) -> str:
        """Add or update patient-level data"""
        patient_id = str(dicom_slice.PatientID)
        patient_info = PatientInfo(
            patient_id=patient_id,
            name=str(dicom_slice.PatientName),
            age=str(dicom_slice.PatientAge),
            sex=str(dicom_slice.PatientSex)
        )
        
        # Update patient_df if new patient
        if patient_id not in self.patient_df['patient_id'].values:
            new_patient = pd.DataFrame([vars(patient_info)])
            self.patient_df = pd.concat([self.patient_df, new_patient], ignore_index=True)
        
        return patient_id

    def add_study_data(self, dicom_slice, patient_id: str) -> str:
        """Add or update study-level data"""
        study_uid = str(dicom_slice.StudyInstanceUID)
        study_info = StudyInfo(
            study_instance_uid=study_uid,
            study_date=str(dicom_slice.StudyDate),
            study_description=str(getattr(dicom_slice, 'StudyDescription', ''))
        )
        
        if study_uid not in self.study_df['study_instance_uid'].values:
            new_study = pd.DataFrame([{
                **vars(study_info),
                'patient_id': patient_id
            }])
            self.study_df = pd.concat([self.study_df, new_study], ignore_index=True)
        
        return study_uid

    def add_series_data(self, dicom_slice, study_uid: str) -> str:
        """Add or update series-level data"""
        series_uid = str(dicom_slice.SeriesInstanceUID)
        series_info = SeriesInfo(
            series_instance_uid=series_uid,
            series_description=str(dicom_slice.SeriesDescription),
            series_number=int(dicom_slice.SeriesNumber),
            slice_thickness=float(getattr(dicom_slice, 'SliceThickness', 0.0)),
            pixel_spacing=tuple(map(float, dicom_slice.PixelSpacing))
        )
        
        if series_uid not in self.series_df['series_instance_uid'].values:
            new_series = pd.DataFrame([{
                **vars(series_info),
                'study_instance_uid': study_uid
            }])
            self.series_df = pd.concat([self.series_df, new_series], ignore_index=True)
        
        return series_uid

    def add_slice(self, dicom_slice, series_uid: str, has_fibroid: Optional[bool] = None):
        """Add a single slice with its pixel data"""
        slice_id = f"{series_uid}_{dicom_slice.InstanceNumber}"
        
        # Add slice metadata
        new_slice = pd.DataFrame([{
            'slice_id': slice_id,
            'series_instance_uid': series_uid,
            'instance_number': int(dicom_slice.InstanceNumber),
            'slice_location': float(getattr(dicom_slice, 'SliceLocation', 0.0)),
            'has_fibroid': has_fibroid
        }])
        self.slice_df = pd.concat([self.slice_df, new_slice], ignore_index=True)
        
        # Store pixel data
        with h5py.File(self.h5_path, 'a') as f:
            if slice_id in f:
                del f[slice_id]
            f.create_dataset(slice_id, data=dicom_slice.pixel_array, compression="gzip", compression_opts=9)

    def add_dicom_file(self, dicom_slice, has_fibroid: Optional[bool] = None):
        """Process a single DICOM file and add to appropriate levels"""
        # Add data at each level of the hierarchy
        patient_id = self.add_patient_data(dicom_slice)
        study_uid = self.add_study_data(dicom_slice, patient_id)
        series_uid = self.add_series_data(dicom_slice, study_uid)
        self.add_slice(dicom_slice, series_uid, has_fibroid)

    def get_patient_series(self, patient_id: str) -> pd.DataFrame:
        """Get all series for a specific patient"""
        patient_studies = self.study_df[self.study_df['patient_id'] == patient_id]
        return self.series_df[self.series_df['study_instance_uid'].isin(patient_studies['study_instance_uid'])]

    def get_series_slices(self, series_uid: str) -> pd.DataFrame:
        """Get all slices for a specific series"""
        return self.slice_df[self.slice_df['series_instance_uid'] == series_uid].sort_values('instance_number')

    def get_slice_pixel_data(self, slice_id: str) -> np.ndarray:
        """Get pixel data for a specific slice"""
        with h5py.File(self.h5_path, 'r') as f:
            return f[slice_id][:]