import matplotlib.pyplot as plt
import time
import numpy as np
import io
import os
from PIL import Image
import cv2
from tqdm import trange, tqdm

import saverloader
import imageio.v2 as imageio
from nets.pips import Pips
import utils.improc
import random
import glob
from utils.basic import print_, print_stats
import torch
from tensorboardX import SummaryWriter
import torch.nn.functional as F

video_path = '/home/vid/hdd/file/project/240-ЕВРАЗ/test/20220909/14_17_04.mp4'
new_video_path = '/home/vid/hdd/file/project/240-ЕВРАЗ/test/20220909/14_17_04_tracker.mp4'
resize_to = (1280, 720)
batch_seconds = 0.2

random.seed(2)
np.random.seed(2)

cap = cv2.VideoCapture(video_path)
cam_fps = int(cap.get(cv2.CAP_PROP_FPS))
totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frames_window = int(cam_fps * batch_seconds)  # seconds

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
new_video = cv2.VideoWriter(new_video_path, fourcc, cam_fps, (1920, 1080))


def get_random_color():
    # randomcolor = (random.randint(50, 200), random.randint(50, 200), random.randint(0, 150))
    randomcolor = (random.randint(0, 150), random.randint(50, 200), random.randint(50, 200))
    return randomcolor


class VideoPointTracker:
    def __init__(self, S=8, stride=8):
        # self.model = Pips(S=S, stride=stride).cuda()
        # _ = saverloader.load('reference_model', self.model)
        # self.model.eval()
        # self.global_step = 0
        self.points = []
        self.local_step = 0

    def _on_click(self, event, x, y, p1, p2):
        if event == cv2.EVENT_LBUTTONDOWN:
            coordinate = (x, y)
            color = get_random_color()
            self.points.append({'coordinate': coordinate, 'color': color})
            cv2.circle(img, coordinate, 5, color, -1)

    def set_points(self, img):
        while True:
            cv2.imshow('set track point', img.copy())
            cv2.setMouseCallback('set track point', tracker._on_click)
            k = cv2.waitKey(cam_fps)
            if k in [32, 27]:  # [Space, Esc] key to stop
                cv2.destroyAllWindows()
                break
        return img

    def tracking(self, rgbs):
        rgbs = torch.Tensor(np.array(rgbs)).permute(0, 3, 1, 2).unsqueeze(0).cuda().float()
        print(rgbs.shape)
        B, S, C, H, W = rgbs.shape

        trajs_e = torch.zeros((B, S, len(self.points), 2), dtype=torch.float32, device='cuda')  # 2 because x and y
        for point_idx, point in enumerate(self.points):
            pass
        exit()


tracker = VideoPointTracker()

p_bar = trange(0, int(totalFrames), colour='green', leave=False)
rgbs = []
for img_idx in p_bar:
    p_bar.set_description(f"{img_idx}")
    ret, img = cap.read()
    if not ret:
        continue
    elif resize_to is not None:
        img = cv2.resize(img, resize_to)
    if tracker.points == []:  # set points if have no points
        tracker.set_points(img)
    if img_idx % frames_window == frames_window - 1 or img_idx == totalFrames - 1:
        rgbs.append(img)
        with torch.no_grad():
            tracker.tracking(rgbs)
    else:
        rgbs.append(img)

cv2.destroyAllWindows()
new_video.release()
