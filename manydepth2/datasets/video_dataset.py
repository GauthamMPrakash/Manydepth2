import os
import cv2
import numpy as np
import PIL.Image as pil
from PIL import Image

from .mono_dataset import MonoDataset


class VideoDataset(MonoDataset):
    """Video dataset which loads frames from a video file sequentially.
    
    This dataset is designed to be a drop-in replacement for KITTIRAWDataset
    and provides frames from a video file in the same format.
    """
    
    def __init__(self, video_path, height, width, frame_idxs, num_scales, 
                 is_train=False, img_ext='.png', K=None, **kwargs):
        """
        Args:
            video_path: Path to the video file
            height: Output height for frames
            width: Output width for frames
            frame_idxs: List of frame indices relative to current frame (e.g., [0, -1, 1])
            num_scales: Number of scales for multi-scale processing
            is_train: Whether this is for training (affects augmentation)
            img_ext: Image extension (kept for compatibility)
            K: Camera intrinsics matrix (4x4). If None, uses default normalized intrinsics
        """
        # Create a dummy filenames list with frame indices
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get total number of frames
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # Create filenames list where each entry is "video_name frame_index l"
        # This mimics the KITTI format: "folder_name frame_index side"
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        filenames = [f"{video_name} {i} l" for i in range(self.total_frames)]
        
        # Initialize parent class
        super(VideoDataset, self).__init__(
            data_path="",  # Not used for video
            filenames=filenames,
            height=height,
            width=width,
            frame_idxs=frame_idxs,
            num_scales=num_scales,
            is_train=is_train,
            img_ext=img_ext,
            **kwargs
        )
        
        # Set camera intrinsics
        if K is not None:
            self.K = K.copy()
        else:
            # Default normalized intrinsics (similar to KITTI)
            self.K = np.array([[0.58, 0, 0.5, 0],
                               [0, 1.92, 0.5, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]], dtype=np.float32)
        
        # Cache for frames to avoid re-reading
        self.frame_cache = {}
        self.cache_size = 100  # Maximum number of frames to cache
        
    def __len__(self):
        return self.total_frames
        
    def __del__(self):
        """Clean up video capture object"""
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
    
    def check_depth(self):
        """Video datasets don't have depth information"""
        return False
    
    def index_to_folder_and_frame_idx(self, index):
        """Convert index to folder name, frame index, and side"""
        line = self.filenames[index].split()
        folder = line[0]  # video name
        frame_index = int(line[1])
        side = line[2] if len(line) > 2 else "l"
        return folder, frame_index, side
    
    def get_image_path(self, folder, frame_index, side):
        """Return a dummy path for compatibility"""
        return f"{self.video_path}:frame_{frame_index}"
    
    def _read_frame(self, frame_index):
        """Read a specific frame from the video"""
        # Check cache first
        if frame_index in self.frame_cache:
            return self.frame_cache[frame_index]
        
        # Ensure frame index is within bounds
        if frame_index < 0 or frame_index >= self.total_frames:
            # Return a black frame for out-of-bounds indices
            return Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8))
        
        # Set video position to the desired frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        
        # Read the frame
        ret, frame = self.cap.read()
        if not ret:
            # Return a black frame if reading fails
            return Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8))
        
        # Convert BGR to RGB
        aspect = 3.33
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h,w,c = frame.shape
        crop_begin = int(h/2-w/aspect/2)
        crop_end = int(h/2 + w/aspect/2)
        frame = frame[crop_begin:crop_end,:]  #Crop the image to same aspect ratio as training image

        # Convert to PIL Image
        pil_image = Image.fromarray(frame)
        
        # Add to cache
        if len(self.frame_cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.frame_cache))
            del self.frame_cache[oldest_key]
        
        self.frame_cache[frame_index] = pil_image
        return pil_image
    
    def get_color(self, folder, frame_index, side, do_flip):
        """Get color image for the specified frame"""
        # Read frame from video
        color = self._read_frame(frame_index)
        
        # Apply horizontal flip if requested
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
        
        return color
    
    def load_intrinsics(self, folder, frame_index):
        """Load camera intrinsics"""
        return self.K.copy()
    
    def get_depth(self, folder, frame_index, side, do_flip):
        """Video datasets don't have depth information"""
        raise NotImplementedError("Video datasets don't have depth information")
    
    def get_video_info(self):
        """Get information about the video"""
        return {
            'total_frames': self.total_frames,
            'fps': self.fps,
            'video_path': self.video_path
        }
    
    def set_frame_position(self, frame_index):
        """Set the current frame position (for sequential reading)"""
        if 0 <= frame_index < self.total_frames:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    
    def get_next_frame(self):
        """Get the next frame in sequence"""
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame)
    
    def reset_video(self):
        """Reset video to the beginning"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.frame_cache.clear()