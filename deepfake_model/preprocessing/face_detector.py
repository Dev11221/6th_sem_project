"""
face_detector.py
Detect and crop faces from frames
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


class FaceDetector:
    """
    Detect faces using Haar Cascade, MTCNN, or RetinaFace
    """
    
    def __init__(self, method: str = 'haar', min_confidence: float = 0.9):
        """
        Initialize face detector
        
        Args:
            method: Detection method ('haar', 'mtcnn', 'retinaface')
            min_confidence: Minimum confidence for detection
        """
        self.method = method.lower()
        self.min_confidence = min_confidence
        
        if self.method == 'haar':
            self._init_haar()
        elif self.method == 'mtcnn':
            self._init_mtcnn()
        elif self.method == 'retinaface':
            self._init_retinaface()
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _init_haar(self):
        """Initialize Haar Cascade"""
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.detector = cv2.CascadeClassifier(cascade_path)
    
    def _init_mtcnn(self):
        """Initialize MTCNN"""
        try:
            from mtcnn import MTCNN
            self.detector = MTCNN()
        except ImportError:
            raise ImportError("Install MTCNN: pip install mtcnn")
    
    def _init_retinaface(self):
        """Initialize RetinaFace"""
        try:
            from retinaface import RetinaFace
            self.detector = RetinaFace
        except ImportError:
            raise ImportError("Install RetinaFace: pip install retina-face")
    
    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in frame
        
        Args:
            frame: Input frame (BGR)
        
        Returns:
            List of bounding boxes [(x, y, w, h), ...]
        """
        if self.method == 'haar':
            return self._detect_haar(frame)
        elif self.method == 'mtcnn':
            return self._detect_mtcnn(frame)
        elif self.method == 'retinaface':
            return self._detect_retinaface(frame)
    
    def _detect_haar(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect using Haar Cascade"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]
    
    def _detect_mtcnn(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect using MTCNN"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = self.detector.detect_faces(rgb)
        
        faces = []
        for det in detections:
            if det['confidence'] >= self.min_confidence:
                x, y, w, h = det['box']
                faces.append((max(0, x), max(0, y), max(0, w), max(0, h)))
        return faces
    
    def _detect_retinaface(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect using RetinaFace"""
        detections = self.detector.detect_faces(frame)
        
        faces = []
        if isinstance(detections, dict):
            for key, det in detections.items():
                if det['score'] >= self.min_confidence:
                    area = det['facial_area']
                    x, y = area[0], area[1]
                    w, h = area[2] - area[0], area[3] - area[1]
                    faces.append((x, y, w, h))
        return faces
    
    def crop_faces(self, frame: np.ndarray, 
                   padding: float = 0.2) -> List[np.ndarray]:
        """
        Detect and crop faces with padding
        
        Args:
            frame: Input frame
            padding: Padding ratio (0.2 = 20% on each side)
        
        Returns:
            List of cropped face images
        """
        faces = self.detect(frame)
        cropped = []
        
        h_img, w_img = frame.shape[:2]
        
        for (x, y, w, h) in faces:
            # Add padding
            pad_w = int(w * padding)
            pad_h = int(h * padding)
            
            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(w_img, x + w + pad_w)
            y2 = min(h_img, y + h + pad_h)
            
            cropped.append(frame[y1:y2, x1:x2])
        
        return cropped
    
    def get_largest_face(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Get largest face from frame
        
        Args:
            frame: Input frame
        
        Returns:
            Cropped largest face or None
        """
        faces = self.detect(frame)
        if not faces:
            return None
        
        # Find largest by area
        largest = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest
        
        return frame[y:y+h, x:x+w]


# ============ USAGE EXAMPLE ============
if __name__ == "__main__":
    detector = FaceDetector(method='haar')
    
    # Test on image
    img = cv2.imread('test.jpg')
    if img is not None:
        faces = detector.detect(img)
        print(f"Detected {len(faces)} faces")
        
        cropped_faces = detector.crop_faces(img, padding=0.2)
        print(f"Cropped {len(cropped_faces)} faces")