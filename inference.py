import torch
import cv2
import numpy as np
import os.path as osp
import time
from lnlface.archs.basenet_arch import LNLFaceNet
import face_alignment  # pip install face-alignment or conda install -c 1adrianb face_alignment
import argparse
import os
import math
from torchvision.transforms.functional import normalize
from tqdm import tqdm
import glob

FaceDetection = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D,
                                             device='cuda' if torch.cuda.is_available() else 'cpu')


def read_img_tensor(img_path=None, return_landmark=True):  # rgb -1~1
    Img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # BGR or G
    if Img.ndim == 2:
        Img = cv2.cvtColor(Img, cv2.COLOR_GRAY2RGB)  # GGG
    else:
        Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)  # RGB

    if Img.shape[0] < 512 or Img.shape[1] < 512:
        Img = cv2.resize(Img, (512, 512), interpolation=cv2.INTER_AREA)

    ImgForLands = Img.copy()
    Img = Img.transpose((2, 0, 1)) / 255.0
    Img = torch.from_numpy(Img).float()
    # normalize(Img, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)
    ImgTensor = Img.unsqueeze(0)
    SelectPred = None
    if return_landmark:
        try:
            PredsAll = FaceDetection.get_landmarks(ImgForLands)
        except:
            print('Error in detecting this face {}. Continue...'.format(img_path))
            PredsAll = None
        if PredsAll is None:
            print('Warning: No face is detected in {}. Continue...'.format(img_path))
            return ImgTensor, None
        ins = 0
        if len(PredsAll) != 1:
            hights = []
            for l in PredsAll:
                hights.append(l[8, 1] - l[19, 1])
            ins = hights.index(max(hights))
            print('Warning: Too many faces are detected, only handle the largest one...')
        SelectPred = PredsAll[ins]
    return ImgTensor, SelectPred


def get_component_location(Landmarks, re_read=False):
    if re_read:
        ReadLandmark = []
        with open(Landmarks, 'r') as f:
            for line in f:
                tmp = [float(i) for i in line.split(' ') if i != '\n']
                ReadLandmark.append(tmp)
        ReadLandmark = np.array(ReadLandmark)  #
        Landmarks = np.reshape(ReadLandmark, [-1, 2])  # 68*2
    Map_LE_B = list(np.hstack((range(17, 22), range(36, 42))))
    Map_RE_B = list(np.hstack((range(22, 27), range(42, 48))))
    Map_LE = list(range(36, 42))
    Map_RE = list(range(42, 48))
    Map_NO = list(range(29, 36))
    Map_MO = list(range(48, 68))

    Landmarks[Landmarks > 504] = 504
    Landmarks[Landmarks < 8] = 8

    # left eye
    Mean_LE = np.mean(Landmarks[Map_LE], 0)
    L_LE1 = Mean_LE[1] - np.min(Landmarks[Map_LE_B, 1])
    L_LE1 = L_LE1 * 1.3
    L_LE2 = L_LE1 / 1.9
    L_LE_xy = L_LE1 + L_LE2
    L_LE_lt = [L_LE_xy / 2, L_LE1]
    L_LE_rb = [L_LE_xy / 2, L_LE2]
    Location_LE = np.hstack((Mean_LE - L_LE_lt + 1, Mean_LE + L_LE_rb)).astype(int)

    # right eye
    Mean_RE = np.mean(Landmarks[Map_RE], 0)
    L_RE1 = Mean_RE[1] - np.min(Landmarks[Map_RE_B, 1])
    L_RE1 = L_RE1 * 1.3
    L_RE2 = L_RE1 / 1.9
    L_RE_xy = L_RE1 + L_RE2
    L_RE_lt = [L_RE_xy / 2, L_RE1]
    L_RE_rb = [L_RE_xy / 2, L_RE2]
    Location_RE = np.hstack((Mean_RE - L_RE_lt + 1, Mean_RE + L_RE_rb)).astype(int)

    # nose
    Mean_NO = np.mean(Landmarks[Map_NO], 0)
    L_NO1 = (np.max([Mean_NO[0] - Landmarks[31][0], Landmarks[35][0] - Mean_NO[0]])) * 1.25
    L_NO2 = (Landmarks[33][1] - Mean_NO[1]) * 1.1
    L_NO_xy = L_NO1 * 2
    L_NO_lt = [L_NO_xy / 2, L_NO_xy - L_NO2]
    L_NO_rb = [L_NO_xy / 2, L_NO2]
    Location_NO = np.hstack((Mean_NO - L_NO_lt + 1, Mean_NO + L_NO_rb)).astype(int)

    # mouth
    Mean_MO = np.mean(Landmarks[Map_MO], 0)
    L_MO = np.max((np.max(np.max(Landmarks[Map_MO], 0) - np.min(Landmarks[Map_MO], 0)) / 2, 16)) * 1.1
    MO_O = Mean_MO - L_MO + 1
    MO_T = Mean_MO + L_MO
    MO_T[MO_T > 510] = 510
    Location_MO = np.hstack((MO_O, MO_T)).astype(int)
    return torch.cat([torch.FloatTensor(Location_LE).unsqueeze(0), torch.FloatTensor(Location_RE).unsqueeze(0),
                      torch.FloatTensor(Location_NO).unsqueeze(0), torch.FloatTensor(Location_MO).unsqueeze(0)], dim=0)


def check_bbox(imgs, boxes):
    boxes = boxes.view(-1, 4, 4)
    imgWithBox = []
    colors = [(0, 255, 0), (0, 255, 0), (255, 255, 0), (255, 0, 0)]
    i = 0
    for img, box in zip(imgs, boxes):
        img = (img + 1) / 2 * 255
        img2 = img.permute(1, 2, 0).float().cpu().flip(2).numpy().copy()
        for idx, point in enumerate(box):
            cv2.rectangle(img2, (int(point[0]), int(point[1])), (int(point[2]), int(point[3])), color=colors[idx],
                          thickness=2)
            img3 = (torch.from_numpy(img2).cuda() / 255. - 0.5) / 0.5
        cv2.imwrite('./ttt_{:02d}.png'.format(i), img2)
        i += 1


if __name__ == '__main__':
    '''
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--img_root', type=str, default="/root/yanwd/dataset/FFHQ_512/*.png",
                        help='input path of lq image/data/yanwd_data/dataset/Testset/xxxxxx/*.png')
    parser.add_argument('-d', '--out_path', type=str, default='/root/yanwd/projects/LNLFace/results/Web/610k/',
                        help='save path of restoration result')
    parser.add_argument('-w', '--weight_path', type=str, default='/root/yanwd/projects/LNLFace/experiments/0818_train_lnlfacev2_hifacegan_gan_nofix/models/net_g_610000.pth',
                        help='weight path of model')
    parser.add_argument('--check', action='store_true',
                        help='save the face images with landmarks shown on them to check the performance')
    args = parser.parse_args()

    os.makedirs(args.out_path, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LNLFaceNet().to(device)  #
    weights = torch.load(args.weight_path)
    model.load_state_dict(weights["params"], strict=True)

    model.eval()
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()

    print('{:>8s} : {}'.format('Using device', device))
    print('{:>8s} : {:.2f}M'.format('Model params', num_params / 1e6))
    torch.cuda.empty_cache()

    for lq_path in glob.glob(args.img_root):
        lq, lq_landmarks = read_img_tensor(lq_path, return_landmark=True)
        if lq_landmarks is None:
            print('Error in detecting landmarks of {}. Maybe its quality is very low. Continue...'.format(lq_path))
            continue

        if args.check:
            Img = cv2.imread(lq_path, cv2.IMREAD_UNCHANGED)
            PathSplits = osp.split(lq_path)
            CheckTmpPath = osp.join(PathSplits[0], 'TmpCheckLQLandmarks')
            os.makedirs(CheckTmpPath, exist_ok=True)
            for point in lq_landmarks[17:, 0:2]:
                cv2.circle(Img, (int(point[0]), int(point[1])), 1, (0, 255, 0), 4)
            cv2.imwrite(osp.join(CheckTmpPath, PathSplits[1]), Img)
            print('Checking detected landmarks in {}'.format(CheckTmpPath))

        start_time = time.time()
        LQLocs = get_component_location(lq_landmarks)

        # check_bbox(lq, LQLocs.unsqueeze(0))

        with torch.no_grad():
            try:
                result = model(lq.to(device), LQLocs.unsqueeze(0))
                end_time = time.time()
                test_time = end_time - start_time
                print(test_time)
            except:
                print('There may be something wrong with the detected component locations. Continue...')
                continue
        # save_generic = GenericResult * 0.5 + 0.5
        save_img = result.squeeze(0).permute(1, 2, 0).flip(2)  # RGB->BGR
        save_img = np.clip(save_img.float().cpu().numpy(), 0, 1) * 255.0

        save_base_name = osp.basename(lq_path).split('.')[0]

        cv2.imwrite(osp.join(args.out_path, save_base_name + '.png'), save_img)

