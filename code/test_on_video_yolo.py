import sys, os, argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
from PIL import Image

import datasets, hopenet, utils

from skimage import io

import PyTorch_YOLOv3.detector as face_detector

CONF_THRESHOLD = 0.5

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--gpu', dest='gpu_id', 
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--pose_weights', 
                        dest='pose_weights', 
                        help='Path of pose estimation model snapshot.',
                        default='/proj/llfr/staff/mmeyn/model-weights/hopenet/hopenet_robust_alpha1.pkl',
                        type=str)
    parser.add_argument('--det_weights', 
                        dest='det_weights',
                        help='Path of face detection model weights.',
                        default='/disk1/mma0448/dev/PyTorch_YOLOv3/weights/yolov3-wider_16000.weights')
    parser.add_argument('--det_model', 
                        default='/disk1/mma0448/dev/PyTorch_YOLOv3/config/yolov3-face.cfg')
    parser.add_argument('--det_meta', 
                        default='/disk1/mma0448/dev/PyTorch_YOLOv3/data/face.names')
    parser.add_argument('--video', dest='video_path', help='Path of video')
    parser.add_argument('--output_string', dest='output_string', help='String appended to output file')
    parser.add_argument('--n_frames', dest='n_frames', help='Number of frames', type=int)
    parser.add_argument('--fps', dest='fps', help='Frames per second of source video', type=float, default=30.)
    parser.add_argument('--output_dir')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True

    batch_size = 1
    gpu = args.gpu_id
    pose_weights = args.pose_weights
    out_dir = args.output_dir
    video_path = args.video_path

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if not os.path.exists(args.video_path):
        sys.exit('Video does not exist')

    # ResNet50 structure
    model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)

    # face detection model
    print('Initializing face detector')
    cnn_face_detector = face_detector.YoloDetector(args.det_model,
                                                   args.det_weights, 
                                                   class_path=args.det_meta,
                                                   gpu=0)

    print ('Loading pose estimator snapshot.')
    # Load snapshot
    saved_state_dict = torch.load(pose_weights)
    model.load_state_dict(saved_state_dict)

    print ('Loading data.')

    transformations = transforms.Compose([transforms.Scale(224),
    transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    model.cuda(gpu)

    print ('Ready to test network.')

    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    total = 0

    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)

    video = cv2.VideoCapture(video_path)

    # New cv2
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    output_string = args.output_string or os.path.splitext(os.path.split(video_path)[-1])[0]
    out_filename = 'output-%s.avi' % output_string
    out_path = os.path.join(out_dir, out_filename)
    out = cv2.VideoWriter(out_path, fourcc, args.fps, (width, height))

    txt_out = open('{}/output-{}.txt'.format(out_dir, output_string), 'w')

    frame_num = 1
    if args.n_frames is None:
        n_frames = float('inf')

    while frame_num <= n_frames:
        print (frame_num)
        try:
            ret,frame = video.read()
            if ret == False:
                break
        except:
            break
        cv2_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        # Dlib detect
        pil_img = Image.fromarray(cv2_frame)
        dets, _ = cnn_face_detector.detect(pil_img)

        for idx, det in enumerate(dets):
            # Get x_min, y_min, x_max, y_max, conf
            x_min, y_min, x_max, y_max, conf, _, _ = det

            if conf > CONF_THRESHOLD:
                bbox_width = abs(x_max - x_min)
                bbox_height = abs(y_max - y_min)
                x_min -= 2 * bbox_width // 4
                x_max += 2 * bbox_width // 4
                y_min -= 3 * bbox_height // 4
                y_max += bbox_height // 4
                x_min = max(x_min, 0); y_min = max(y_min, 0)
                x_max = min(frame.shape[1], x_max); y_max = min(frame.shape[0], y_max)
                # Crop image
                x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)
                img = cv2_frame[y_min:y_max,x_min:x_max]
                img = Image.fromarray(img)

                # Transform
                img = transformations(img)
                img_shape = img.size()
                img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
                img = Variable(img).cuda(gpu)

                yaw, pitch, roll = model(img)

                yaw_predicted = F.softmax(yaw)
                pitch_predicted = F.softmax(pitch)
                roll_predicted = F.softmax(roll)
                # Get continuous predictions in degrees.
                yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 99
                pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 99
                roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor) * 3 - 99

                # Print new frame with cube and axis
                txt_out.write(str(frame_num) + ' %f %f %f\n' % (yaw_predicted, pitch_predicted, roll_predicted))
                # utils.plot_pose_cube(frame, yaw_predicted, pitch_predicted, roll_predicted, (x_min + x_max) / 2, (y_min + y_max) / 2, size = bbox_width)
                utils.draw_axis(frame, yaw_predicted, pitch_predicted, roll_predicted, tdx = (x_min + x_max) / 2, tdy= (y_min + y_max) / 2, size = bbox_height/2)
                # Plot expanded bounding box
                # cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 1)

        out.write(frame)
        frame_num += 1

    out.release()
    video.release()
