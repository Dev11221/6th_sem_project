import cv2
test_video_path = "D:\\PROJECTS\\6th_sem_project\\Hollow Knight Silksong 2025-12-27 23-14-16.mp4"
# load video from a file path
class VideoReader:
    def __init__(self,video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Error opening video file: {video_path}")
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(self.frame_count, self.fps)
VideoReader_instance = VideoReader(test_video_path)