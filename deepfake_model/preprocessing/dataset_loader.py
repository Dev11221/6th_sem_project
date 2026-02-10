"""
dataset_loaders.py
Load video paths and labels from different dataset formats
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict


class DFDCLoader:
    """
    DFDC Dataset Loader
    Structure: metadata.json contains labels
    """
    
    def __init__(self, dataset_path: str):
        """
        Args:
            dataset_path: Path to DFDC dataset folder
        """
        self.dataset_path = dataset_path
        self.metadata_file = os.path.join(dataset_path, 'metadata.json')
    
    def load_videos(self, max_videos: int = None) -> Tuple[List[str], List[int]]:
        """
        Load DFDC video paths and labels
        
        Returns:
            (video_paths, labels) where label: 0=REAL, 1=FAKE
        """
        if not os.path.exists(self.metadata_file):
            raise FileNotFoundError(f"Metadata not found: {self.metadata_file}")
        
        with open(self.metadata_file, 'r') as f:
            metadata = json.load(f)
        
        video_paths = []
        labels = []
        
        for video_name, info in metadata.items():
            video_path = os.path.join(self.dataset_path, video_name)
            
            if os.path.exists(video_path):
                # DFDC: "label" key with "FAKE" or "REAL"
                label = 1 if info.get('label') == 'FAKE' else 0
                
                video_paths.append(video_path)
                labels.append(label)
                
                if max_videos and len(video_paths) >= max_videos:
                    break
        
        print(f"DFDC: Loaded {len(video_paths)} videos")
        print(f"  Real: {labels.count(0)}, Fake: {labels.count(1)}")
        
        return video_paths, labels


class FaceForensicsLoader:
    """
    FaceForensics++ Dataset Loader
    Structure: 
        - original_sequences/youtube/c23/videos/ (real)
        - manipulated_sequences/{method}/c23/videos/ (fake)
    Methods: Deepfakes, Face2Face, FaceSwap, NeuralTextures
    """
    
    def __init__(self, dataset_path: str, compression: str = 'c23'):
        """
        Args:
            dataset_path: Path to FF++ root
            compression: 'c0' (raw), 'c23' (light), 'c40' (heavy)
        """
        self.dataset_path = dataset_path
        self.compression = compression
        self.methods = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
    
    def load_videos(self, 
                   include_methods: List[str] = None,
                   max_videos_per_class: int = None) -> Tuple[List[str], List[int]]:
        """
        Load FF++ videos
        
        Args:
            include_methods: Which manipulation methods to include (None = all)
            max_videos_per_class: Max videos per real/fake category
        
        Returns:
            (video_paths, labels)
        """
        if include_methods is None:
            include_methods = self.methods
        
        video_paths = []
        labels = []
        
        # Load REAL videos
        real_path = os.path.join(
            self.dataset_path, 
            'original_sequences', 'youtube', self.compression, 'videos'
        )
        
        if os.path.exists(real_path):
            real_videos = [
                os.path.join(real_path, f) 
                for f in os.listdir(real_path) 
                if f.endswith('.mp4')
            ]
            
            if max_videos_per_class:
                real_videos = real_videos[:max_videos_per_class]
            
            video_paths.extend(real_videos)
            labels.extend([0] * len(real_videos))
        
        # Load FAKE videos
        for method in include_methods:
            fake_path = os.path.join(
                self.dataset_path,
                'manipulated_sequences', method, self.compression, 'videos'
            )
            
            if os.path.exists(fake_path):
                fake_videos = [
                    os.path.join(fake_path, f)
                    for f in os.listdir(fake_path)
                    if f.endswith('.mp4')
                ]
                
                if max_videos_per_class:
                    fake_videos = fake_videos[:max_videos_per_class]
                
                video_paths.extend(fake_videos)
                labels.extend([1] * len(fake_videos))
        
        print(f"FF++: Loaded {len(video_paths)} videos")
        print(f"  Real: {labels.count(0)}, Fake: {labels.count(1)}")
        
        return video_paths, labels


class CelebDFLoader:
    """
    Celeb-DF-v2 Dataset Loader
    Structure:
        - Celeb-real/ (real videos)
        - Celeb-synthesis/ (fake videos)
        - List_of_testing_videos.txt (test split info)
    """
    
    def __init__(self, dataset_path: str):
        """
        Args:
            dataset_path: Path to Celeb-DF-v2 root
        """
        self.dataset_path = dataset_path
        self.real_path = os.path.join(dataset_path, 'Celeb-real')
        self.fake_path = os.path.join(dataset_path, 'Celeb-synthesis')
    
    def load_videos(self, 
                   max_real: int = None,
                   max_fake: int = None) -> Tuple[List[str], List[int]]:
        """
        Load Celeb-DF videos
        
        Args:
            max_real: Max real videos
            max_fake: Max fake videos
        
        Returns:
            (video_paths, labels)
        """
        video_paths = []
        labels = []
        
        # Load REAL videos
        if os.path.exists(self.real_path):
            real_videos = [
                os.path.join(self.real_path, f)
                for f in os.listdir(self.real_path)
                if f.endswith('.mp4')
            ]
            
            if max_real:
                real_videos = real_videos[:max_real]
            
            video_paths.extend(real_videos)
            labels.extend([0] * len(real_videos))
        
        # Load FAKE videos
        if os.path.exists(self.fake_path):
            fake_videos = [
                os.path.join(self.fake_path, f)
                for f in os.listdir(self.fake_path)
                if f.endswith('.mp4')
            ]
            
            if max_fake:
                fake_videos = fake_videos[:max_fake]
            
            video_paths.extend(fake_videos)
            labels.extend([1] * len(fake_videos))
        
        print(f"Celeb-DF: Loaded {len(video_paths)} videos")
        print(f"  Real: {labels.count(0)}, Fake: {labels.count(1)}")
        
        return video_paths, labels
    
    def load_with_official_split(self) -> Dict[str, Tuple[List[str], List[int]]]:
        """
        Load using official train/test split
        
        Returns:
            {'train': (paths, labels), 'test': (paths, labels)}
        """
        test_list_file = os.path.join(self.dataset_path, 'List_of_testing_videos.txt')
        
        if not os.path.exists(test_list_file):
            raise FileNotFoundError("Official test split file not found")
        
        # Read test video names
        with open(test_list_file, 'r') as f:
            test_videos = set(line.strip() for line in f if line.strip())
        
        # Get all videos
        all_paths, all_labels = self.load_videos()
        
        # Split into train/test
        train_paths, train_labels = [], []
        test_paths, test_labels = [], []
        
        for path, label in zip(all_paths, all_labels):
            video_name = os.path.basename(path)
            
            if video_name in test_videos:
                test_paths.append(path)
                test_labels.append(label)
            else:
                train_paths.append(path)
                train_labels.append(label)
        
        print(f"Official Split:")
        print(f"  Train: {len(train_paths)} videos")
        print(f"  Test: {len(test_paths)} videos")
        
        return {
            'train': (train_paths, train_labels),
            'test': (test_paths, test_labels)
        }


# ============ USAGE EXAMPLE ============
if __name__ == "__main__":
    # # DFDC
    # dfdc = DFDCLoader('deepfake_model/dataset/dfdc')
    # videos, labels = dfdc.load_videos(max_videos=10)
    
    # FaceForensics++
    ff = FaceForensicsLoader('deepfake_model/dataset/ff')
    videos, labels = ff.load_videos(
        include_methods=['Deepfakes', 'Face2Face'],
        max_videos_per_class=5
    )
    
    # Celeb-DF
    celeb = CelebDFLoader('deepfake_model/dataset/celeb-df')
    videos, labels = celeb.load_videos(max_real=20, max_fake=10)
    
    # Celeb-DF with official split
    splits = celeb.load_with_official_split()
    train_videos, train_labels = splits['train']
    test_videos, test_labels = splits['test']