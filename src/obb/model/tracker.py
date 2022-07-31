import numpy as np
import torch
from torchvision.transforms import ToTensor
from torchvision.io import read_video, write_video
from PIL import Image, ImageDraw
from skimage import draw
from random import randint

from obb.model.custom_model import *
from obb.utils.box_ops import *
from obb.utils.infer_ops import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class OBBTracker:
    NUM_CLASSES = 15
    NUM_OFFSETS = 9
    MAX_STRIDE = 32

    def __init__(self, video_path, state_path, write_path):
        self.video_path = video_path
        self.state_path = state_path
        self.write_path = write_path

        # Read video frames from file
        self.frames, _, self.metadata = read_video(filename=self.video_path)
        self.fps = self.metadata['video_fps']
        self.frames = self.frames.float()

        # Truncate frames s.t. their dimensions will be integer multiples of the maximal stride
        frame_height, frame_width = self.frames.shape[1:3]
        self.frames = self.frames[:, :(frame_height // OBBTracker.MAX_STRIDE) * OBBTracker.MAX_STRIDE,
                      :(frame_width // OBBTracker.MAX_STRIDE) * OBBTracker.MAX_STRIDE, :]
        self.frame_height, self.frame_width = self.frames.shape[1:3]

        # Initialize model and load pretrained weights
        state_dict = torch.load(self.state_path, map_location=device)
        self.model = DetectionModel().to(device)
        self.model.load_state_dict(state_dict['state_dict'])

    def simple_detect(self):
        """
        This function simply detects objects in each frame, without tracking. Used for testing purposes only.
        """

        # Create tensor for new video
        frames_pred = self.frames.clone()

        for frame_idx, frame in enumerate(self.frames):
            print(f'Frame #{frame_idx + 1}')

            # Get model prediction
            to_tensor = ToTensor()
            frame_torch = to_tensor(frame.numpy().astype(np.uint8)).unsqueeze(dim=0)
            classification, rep_points = infer(self.model, frame_torch, cls_thr=0.1)
            classification_hard = torch.argmax(classification, dim=1) + 1
            print(f'Number of predictions: {len(classification)}')

            # Convert to Gaussian and then to OBB
            mu, S = rep_points_to_gaussian(rep_points.reshape(-1, OBBTracker.NUM_OFFSETS, 2))
            rects_xywha = gaussian_to_xywha(mu, S)
            rects = xywha_to_xyxy(rects_xywha).reshape(-1, 4, 2)

            # Draw OBBs on current frame
            for rect_idx, rect in enumerate(rects):
                if classification_hard[rect_idx] == 5 or classification_hard[rect_idx] == 6:
                    rect = rect.detach().numpy()
                    for i in range(4):
                        line_coord_rows, line_coord_cols = draw.line(int(rect[i, 1]), int(rect[i, 0]),
                                                                     int(rect[(i + 1) % 4, 1]),
                                                                     int(rect[(i + 1) % 4, 0]))
                        if ((line_coord_rows >= 0).all()
                                and (line_coord_rows < self.frame_height).all()
                                and (line_coord_cols >= 0).all()
                                and (line_coord_cols < self.frame_width).all()):
                            frames_pred[frame_idx, line_coord_rows, line_coord_cols] = torch.Tensor([255, 255, 0])

        # Save video with OBBs
        write_video(filename=self.write_path,
                    video_array=frames_pred,
                    fps=self.fps,
                    video_codec='h264')


if __name__ == '__main__':
    tracker = OBBTracker(video_path='../../../assets/DOTA_sample_data/videos/vid_1.mp4',
                         state_path='./checkpoints_mega/epoch_20.pt',
                         write_path='../../../assets/DOTA_sample_data/videos/vid_pred_1.mp4')

    tracker.simple_detect()
