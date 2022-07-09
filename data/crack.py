from typing import Tuple, List
import numpy as np
import os
from skimage.io import imread
from skimage.transform import resize
from pathlib import Path


class Crack(object):

    def __init__(self, data_root: Path, seq_len: int = 20, image_size: Tuple[int, int] = (256, 132)):
        self.crack_video_frames: List[str] = sorted(os.listdir(data_root))
        self.data_root: Path = data_root
        self.seq_len: int = seq_len
        self.image_size: Tuple[int, int] = image_size

    def __len__(self):
        return len(self.crack_video_frames)

    def __getitem__(self, index):
        image_frames = []

        for frame_idx, img_path in enumerate(sorted((Path(self.data_root) /
                                                     self.crack_video_frames[index]).glob('*.jpg'))):
            image_frame = imread(str(img_path))
            image_frame_resized = resize(image_frame, output_shape=self.image_size)
            image_frames.append(image_frame_resized)

        image_frames_sub_sampled = self._subsample_images(image_frames[1:])
        return image_frames_sub_sampled

    def _subsample_images(self, image_frames: List[np.ndarray]) -> np.ndarray:
        image_frames_np = np.zeros((self.seq_len, self.image_size[0], self.image_size[1], 3),
                                   dtype=np.float32)
        stride = int(np.ceil(len(image_frames) / self.seq_len))
        image_frames_sub = image_frames[::stride]
        if len(image_frames_sub) < self.seq_len:
            image_frames_sub.append(image_frames[-1])
        assert len(image_frames_sub) == self.seq_len, f"After subsampling length of image frame should be equal to " \
                                                      f"the sequence length got image_frames={len(image_frames_sub)}" \
                                                      f" for seq_len={self.seq_len}"
        for img_idx, img in enumerate(image_frames_sub):
            image_frames_np[img_idx] = img / np.max(img)
        return image_frames_np
