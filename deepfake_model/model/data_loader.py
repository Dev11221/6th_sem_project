"""
data_loader.py
Load data in multiple formats for different models/frameworks
"""

import numpy as np
import os
from typing import Tuple


class DataLoader:
    """Load preprocessed data in various formats"""
    
    @staticmethod
    def load_numpy(data_path: str):
        """Load .npy files"""
        X_train = np.load(os.path.join(data_path, 'X_train.npy'))
        X_test = np.load(os.path.join(data_path, 'X_test.npy'))
        y_train = np.load(os.path.join(data_path, 'y_train.npy'))
        y_test = np.load(os.path.join(data_path, 'y_test.npy'))
        
        return X_train, X_test, y_train, y_test
    
    @staticmethod
    def export_to_pytorch_format(data_path: str, output_path: str):
        """Export to PyTorch .pt format"""
        try:
            import torch
            
            X_train, X_test, y_train, y_test = DataLoader.load_numpy(data_path)
            
            # Convert to PyTorch tensors (NHWC -> NCHW)
            torch_data = {
                'X_train': torch.from_numpy(X_train).permute(0, 3, 1, 2).float(),
                'X_test': torch.from_numpy(X_test).permute(0, 3, 1, 2).float(),
                'y_train': torch.from_numpy(y_train).long(),
                'y_test': torch.from_numpy(y_test).long()
            }
            
            os.makedirs(output_path, exist_ok=True)
            torch.save(torch_data, os.path.join(output_path, 'dataset.pt'))
            
            print(f" PyTorch format: {output_path}/dataset.pt")
            print(f"   X_train: {torch_data['X_train'].shape}")
            print(f"   X_test: {torch_data['X_test'].shape}")
        except ImportError:
            print(" Install PyTorch: pip install torch")
    
    @staticmethod
    def get_data_info(data_path: str):
        """Print dataset info"""
        X_train, X_test, y_train, y_test = DataLoader.load_numpy(data_path)
        
        print("="*60)
        print("DATASET INFO")
        print("="*60)
        print(f"X_train: {X_train.shape}")
        print(f"X_test:  {X_test.shape}")
        print(f"Train - Real: {np.sum(y_train==0)}, Fake: {np.sum(y_train==1)}")
        print(f"Test  - Real: {np.sum(y_test==0)}, Fake: {np.sum(y_test==1)}")
        print("="*60)


if __name__ == "__main__":
    # Show info
    DataLoader.get_data_info('deepfake_model/dataset/splits')
    
    # Export to PyTorch
    DataLoader.export_to_pytorch_format(
        'deepfake_model/dataset/splits',
        'deepfake_model/dataset/pytorch_format'
    )