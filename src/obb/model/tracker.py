# Credit to SORT template: https://github.com/abewley/sort

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torchvision.transforms import ToTensor
from torchvision.io import read_video, write_video
from PIL import Image, ImageDraw
from skimage import draw
from random import randint
from filterpy.kalman import KalmanFilter

from obb.model.custom_model import *
from obb.utils.box_ops import *
from obb.utils.infer_ops import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class KalmanTracker:
    """
    A Kalman filter based tracker of a single object.
    The state vector is given as (x, y, vx, vy, theta, omega), and updated by the measurement vector
    (x, y, theta)
    """

    count = 0

    def __init__(self, x0, y0, w0, h0, a0, mu, S):
        """
        Initializes tracker based on initial position and angle.
        """
        self.kf = KalmanFilter(dim_x=6, dim_z=3)

        # State transition matrix
        self.kf.F = np.array([[1, 0, 1, 0, 0, 0],
                              [0, 1, 0, 1, 0, 0],
                              [0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 1, 1],
                              [0, 0, 0, 0, 0, 1]])

        # Measurement function
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0]])

        # Measurement uncertainty
        self.kf.R[2, 2] *= 1e-4

        # Process uncertainty
        self.kf.Q[2:4, 2:4] *= 0.01
        self.kf.Q[-1, -1] *= 1e-4

        # Covariance matrix
        self.kf.P[2:4, 2:4] *= 1000.  # Give high uncertainty to the unobservable initial velocities
        self.kf.P[-1, -1] *= 0.1
        self.kf.P *= 10.

        # Initialize state vector
        self.kf.x[0] = x0
        self.kf.x[1] = y0
        self.kf.x[4] = a0

        # Initialize width and height
        self.width = w0
        self.height = h0

        # Initialize Gaussian distribution parameters
        self.mu = mu
        self.S = S

        # Initialize remaining parameters
        self.time_since_update = 0
        self.id = KalmanTracker.count
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.accepted = False
        color_hex = '%06X' % randint(0, 0xFFFFFF)
        self.color = torch.Tensor([int(color_hex[i:i+2], 16) for i in (0, 2, 4)])

        # Update counter
        KalmanTracker.count += 1

    def update(self, x, y, a):
        """
        Updates state vector based on current detection.
        """
        a_cont = KalmanTracker.shift_angle(self.kf.x[4][0], a)  # Remove angle discontinuity
        # self.color = torch.Tensor([0, 0, 255])
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(np.array([x, y, a_cont]))
        c, s = np.cos(self.kf.x[4][0]), np.sin(self.kf.x[4][0])
        self.mu, self.S = xywha_to_gaussian(
            torch.Tensor([self.kf.x[0][0], self.kf.x[1][0], self.width, self.height, c, s]).unsqueeze(0))
        self.mu = self.mu.squeeze(0)
        self.S = self.S.squeeze(0)

    @staticmethod
    def shift_angle(a1, a2):
        """
        Makes transition between angles continuous by removing multiples of 2*pi.
        """
        return np.unwrap(np.array([a1, a2]))[1]

    def predict(self):
        """
        Advances state vector and returns the predicted position and angle.
        """
        self.kf.predict()
        # self.color = torch.Tensor([0, 255, 0])
        c, s = np.cos(self.kf.x[4][0]), np.sin(self.kf.x[4][0])

        self.mu, self.S = xywha_to_gaussian(
            torch.Tensor([self.kf.x[0][0], self.kf.x[1][0], self.width, self.height, c, s]).unsqueeze(0))
        self.mu = self.mu.squeeze(0)
        self.S = self.S.squeeze(0)
        self.age += 1

        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.kf.x)

        return self.history[-1]


class SORTTracker:
    """
    A multiple object tracker based on the SORT algorithm.
    """

    def __init__(self, max_age=3, min_hits=3, kl_threshold=10):
        """
        Initializes tracker parameters.
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.kl_threshold = kl_threshold
        self.trackers = []
        self.frame_count = 0

    @staticmethod
    def compute_kl_matrix(det_mu, det_S, trk_mu, trk_S):
        """
        Computes KL divergence between each detection and tracked object
        """
        num_det, num_trk = len(det_mu), len(trk_mu)
        kl_mat_flattened = kl_divergence_gaussian(trk_mu.repeat(num_det, 1),
                                                  trk_S.repeat(num_det, 1, 1),
                                                  det_mu.repeat_interleave(num_trk, dim=0),
                                                  det_S.repeat_interleave(num_trk, dim=0), batched=True)
        kl_mat = kl_mat_flattened.reshape(num_det, num_trk)

        return kl_mat

    @staticmethod
    def linear_assignment_zip(cost_mat):
        """
        Performs linear assignment and zips row and column indices.
        """
        row_indices, col_indices = linear_sum_assignment(cost_mat)
        return np.array(list(zip(row_indices, col_indices)))

    def associate_detections_to_trackers(self, det_mu, det_S, kl_threshold=0.5):
        """
        Assigns detections to the tracked objects.
        Returns 3 lists: matches, unmatched_detections, unmatched_trackers
        """
        if len(self.trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(det_mu)), np.empty((0, 5), dtype=int)

        trk_mu, trk_S = torch.stack([trk.mu for trk in self.trackers]), torch.stack([trk.S for trk in self.trackers])
        kl_mat = SORTTracker.compute_kl_matrix(det_mu, det_S, trk_mu, trk_S)

        # Assign each detection to the tracker yielding minimal KL divergence
        if min(kl_mat.shape) > 0:
            a = (kl_mat < kl_threshold)
            if a.sum(axis=1).max() == 1 and a.sum(axis=0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                matched_indices = SORTTracker.linear_assignment_zip(kl_mat)
        else:
            matched_indices = np.empty(shape=(0, 2))

        # Store unmatched detections and trackers
        unmatched_detections = []
        for d in range(len(det_mu)):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)
        unmatched_trackers = []
        for t in range(len(trk_mu)):
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)

        # Filter out matched indices with high KL divergence
        matches = []
        for m in matched_indices:
            if kl_mat[m[0], m[1]] > kl_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

    def update(self, det_mu, det_S):
        """
        Updates trackers based on recent detections.
        """
        self.frame_count += 1

        # Delete trackers yielding NaN values
        to_del = []
        ret = []
        for t in range(len(self.trackers)):
            x = self.trackers[t].predict()
            if np.any(np.isnan(x)):
                to_del.append(t)
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(det_mu, det_S, self.kl_threshold)

        # Convert Gaussian distribution parameters to (x, y, width, height, angle)
        det_xywha = gaussian_to_xywha(det_mu, det_S)
        x, y, w, h, c, s = det_xywha[:, 0], det_xywha[:, 1], det_xywha[:, 2], det_xywha[:, 3], det_xywha[:, 4], det_xywha[:, 5]
        a = cs_to_angle(c, s)

        # Update matched trackers with assigned detections
        for m in matched:
            det_ind, trk_ind = m[0], m[1]
            self.trackers[trk_ind].update(x[det_ind], y[det_ind], a[det_ind])
            if self.trackers[trk_ind].hit_streak >= self.min_hits:
                self.trackers[trk_ind].accepted = True

        # Create and initialize new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanTracker(x[i], y[i], w[i], h[i], a[i], det_mu[i], det_S[i])
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            if trk.accepted:
                ret.append(trk)
            i -= 1
            # Remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        return ret


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

        # Initialize SORT object tracker
        self.sort_tracker = SORTTracker()

        # Initialize model and load pretrained weights
        state_dict = torch.load(self.state_path, map_location=device)
        self.model = DetectionModel().to(device)
        self.model.load_state_dict(state_dict['state_dict'])

    def draw_rect(self, rect, frames, frame_idx, color):
        """
        Adds rectangular patch to frame.
        """
        for i in range(4):
            line_coord_rows, line_coord_cols = draw.line(int(rect[i, 1]), int(rect[i, 0]),
                                                         int(rect[(i + 1) % 4, 1]),
                                                         int(rect[(i + 1) % 4, 0]))
            if ((line_coord_rows >= 0).all()
                    and (line_coord_rows < self.frame_height).all()
                    and (line_coord_cols >= 0).all()
                    and (line_coord_cols < self.frame_width).all()):
                frames[frame_idx, line_coord_rows, line_coord_cols] = color

    def simple_detect(self):
        """
        This function simply detects objects in each frame, without tracking. Used for testing purposes only.
        """

        # Poll random color
        color_hex = '%06X' % randint(0, 0xFFFFFF)
        color_rgb = torch.Tensor([int(color_hex[i:i+2], 16) for i in (0, 2, 4)])

        # Create tensor for new video
        frames_pred = self.frames.clone()

        for frame_idx, frame in enumerate(self.frames):
            if frame_idx == 1:
                break
            print(f'Frame #{frame_idx + 1}')

            # Get model prediction
            to_tensor = ToTensor()
            frame_torch = to_tensor(frame.numpy().astype(np.uint8)).unsqueeze(dim=0)
            classification, rep_points = infer(self.model, frame_torch, cls_thr=0.5)
            classification_hard = torch.argmax(classification, dim=1) + 1
            print(f'Number of predictions: {len(classification)}')

            # Convert to Gaussian and then to OBB
            mu, S = rep_points_to_gaussian(rep_points.reshape(-1, OBBTracker.NUM_OFFSETS, 2))
            rects_xywha = gaussian_to_xywha(mu, S)
            rects = xywha_to_xyxy(rects_xywha).reshape(-1, 4, 2)

            # Draw OBBs on current frame
            for rect_idx, rect in enumerate(rects):
                if classification_hard[rect_idx] == 5 or classification_hard[rect_idx] == 6:
                    self.draw_rect(rect.detach().numpy(), frames_pred, frame_idx, color_rgb)

        # Save video with OBBs
        write_video(filename=self.write_path,
                    video_array=frames_pred,
                    fps=self.fps,
                    video_codec='h264')

    def track_video(self):
        """
        Tracks objects in the given video.
        """

        # Create tensor for new video
        frames_pred = self.frames.clone()

        for frame_idx, frame in enumerate(self.frames):

            print(f'Frame #{frame_idx + 1}')

            # Get model prediction
            to_tensor = ToTensor()
            frame_torch = to_tensor(frame.numpy().astype(np.uint8)).unsqueeze(dim=0)
            classification, rep_points = infer(self.model, frame_torch, cls_thr=0.5)
            classification_hard = torch.argmax(classification, dim=1) + 1
            print(f'Number of predictions: {len(classification)}')

            # Convert to Gaussian representation only object classifies as small or large vehicle
            slv_idx = torch.where(torch.logical_or(classification_hard == 5, classification_hard == 6))
            rep_points_slv = rep_points[slv_idx]
            mu, S = rep_points_to_gaussian(rep_points_slv.reshape(-1, OBBTracker.NUM_OFFSETS, 2))

            # Update object tracker and retrieve tracked objects
            trackers = self.sort_tracker.update(mu.detach(), S.detach())

            for trk in trackers:
                # Convert tracker state into OBB
                x, y, _, _, a, _ = trk.kf.x
                w, h = trk.width, trk.height
                c, s = np.cos(a), np.sin(a)
                rect = xywha_to_xyxy(torch.Tensor([x[0], y[0], w, h, c, s]).unsqueeze(0)).reshape(4, 2)

                # Draw OBB on current frame
                self.draw_rect(rect.detach().numpy(), frames_pred, frame_idx, trk.color)

        # Save video with OBBs
        write_video(filename=self.write_path,
                    video_array=frames_pred,
                    fps=self.fps,
                    video_codec='h264')


if __name__ == '__main__':
    video_num = 6
    video_format = 'mp4'
    epoch = 40

    tracker = OBBTracker(video_path=f'../../../assets/DOTA_sample_data/videos/vid_{video_num}.{video_format}',
                         state_path=f'./checkpoints_final/epoch_{epoch}.pt',
                         write_path=f'../../../assets/DOTA_sample_data/videos/vid_{video_num}_pred.{video_format}')

    tracker.track_video()
