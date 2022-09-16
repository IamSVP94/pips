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

random.seed(125)
np.random.seed(125)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# video_path = '/home/vid/hdd/file/project/RUS_AGRO/video/mini/mini1.mp4'
# new_video_path = '/home/vid/hdd/file/project/RUS_AGRO/video/mini/mini_points.mp4'
# points = [(760, 1000)]


# video_path = '/home/vid/hdd/file/project/RUS_AGRO/video/mini/mini2.mp4'
# new_video_path = '/home/vid/hdd/file/project/RUS_AGRO/video/mini/mini2_points.mp4'
# points = [(960, 650)]

# video_path = '/home/vid/Downloads/video/Очередь.mp4'
# new_video_path = '/home/vid/Downloads/video/Очередь_tracker.mp4'
# points = [(790, 50)]

video_path = '/home/vid/hdd/file/project/240-ЕВРАЗ/test/20220909/14_17_04.mp4'
new_video_path = '/home/vid/hdd/file/project/240-ЕВРАЗ/test/20220909/14_17_04_tracker.mp4'
points = [(1310, 600)]

cap = cv2.VideoCapture(video_path)
cam_fps = cap.get(cv2.CAP_PROP_FPS)
totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

new_video = cv2.VideoWriter(new_video_path, fourcc, cam_fps, (1920, 1080))

stride = 8
batch_seconds = 9
model = Pips(S=8, stride=stride).cuda()
parameters = list(model.parameters())
_ = saverloader.load('reference_model', model)
global_step = 0
model.eval()

p_bar = trange(0, int(totalFrames), colour='green', leave=False)
rgbs = []
# points = [(265, 328)]
colors = [(0, 255, 0), (0, 0, 255)]
frames_window = int(cam_fps * batch_seconds)  # seconds

# TODO: add batch and several points
local_step = 0
for img_idx in p_bar:
    p_bar.set_description(f"{img_idx}")
    ret, img = cap.read()
    if not ret:
        continue

    # cv2.circle(img, points[0], 5, colors[0], -1)
    # img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (640, 360)).astype(int)
    # img = cv2.resize(img, (640, 360)).astype(int)
    # plt.imshow(img)
    # plt.show()
    # exit()
    if img_idx % frames_window == frames_window - 1 or img_idx == totalFrames - 1:
        rgbs.append(img)
        with torch.no_grad():
            rgbs = torch.Tensor(np.array(rgbs)).permute(0, 3, 1, 2).unsqueeze(0).cuda().float()
            B, S, C, H, W = rgbs.shape

            trajs_e = torch.zeros((B, S, len(points), 2), dtype=torch.float32, device='cuda')  # 2 because x and y
            for point_idx, point in enumerate(points):
                if local_step != 0:
                    point = last_track_point
                print(point_idx, local_step, point)
                cur_frame = 0
                done = False
                feat_init = None

                traj_e = torch.zeros((B, S, 2), dtype=torch.float32, device='cuda')
                traj_e[:, 0] = torch.Tensor(point).unsqueeze(0).unsqueeze(0)  # B, 1, 2  # set first position

                while not done:
                    end_frame = cur_frame + stride
                    rgb_seq = rgbs[:, cur_frame:end_frame]  # взяли окно из батча
                    S_local = rgb_seq.shape[1]
                    # последнее окно если меньше чем stride
                    rgb_seq = torch.cat([rgb_seq, rgb_seq[:, -1].unsqueeze(1).repeat(1, stride - S_local, 1, 1, 1)],
                                        dim=1)

                    xys_seq = traj_e[:, cur_frame].reshape(1, -1, 2)  # cur_frame - селектор из тензора
                    outs = model(
                        xys=xys_seq,
                        rgbs=rgb_seq,
                        iters=6,
                        feat_init=feat_init,  # холодный старт или нет
                        return_feat=True)  # coord_predictions, coord_predictions2, vis_e, ffeat, losses
                    preds, _, vis, feat_init, losses = outs  # vis (B, S, 1)
                    # print(outs[0], len(outs[0]))
                    # print('*' * 50)
                    # print(outs[1], len(outs[1]))
                    vis = torch.sigmoid(vis)  # visibility confidence
                    xys = preds[-1].reshape(1, stride, 2)
                    traj_e[:, cur_frame:end_frame] = xys[:, :S_local]

                    found_skip = False
                    thr = 0.9
                    si_last = stride - 1  # last frame we are willing to take
                    si_earliest = 1  # earliest frame we are willing to take
                    si = si_last
                    while not found_skip:  # блок для нахождения перекрытия точки (возможно)
                        if vis[0, si] > thr:
                            # print(f'si={si}, thr={thr}')
                            found_skip = True
                        else:
                            si -= 1
                        if si == si_earliest:
                            thr_pred = thr
                            thr -= 0.02
                            # print(f' decreasing thresh from {thr_pred} to {thr}')
                            si = si_last
                        # print(f'found skip at frame {si}, where we have {vis[0, si].detach().item()}')

                    cur_frame = cur_frame + si

                    if cur_frame >= S:
                        done = True

                trajs_e[:, :, point_idx] = traj_e

        coordinates = trajs_e.squeeze().to(int).cpu().detach().numpy()
        images = rgbs.squeeze().permute(0, 2, 3, 1).cpu().detach().numpy()
        for im_idx, im in enumerate(images):
            im = np.uint8(im)
            cv2.circle(im, coordinates[im_idx], 5, colors[point_idx], -1)
            new_video.write(np.uint8(im))
        else:
            last_track_point = tuple(coordinates[im_idx])
            local_step += 1

        rgbs = []
    else:
        rgbs.append(img)

cv2.destroyAllWindows()
new_video.release()
