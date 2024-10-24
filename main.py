from yolov5.utils.general import(
    check_img_size, non_max_suppression, scale_coords, xyxy2xywh)
from yolov5.utils.torch_utils import select_device, time_synchronized
from yolov5.utils.datasets import letterbox

from utils_ds.parser import get_config
from utils_ds.draw import draw_boxes
from deep_sort import build_tracker

import argparse
import os
import time
import numpy as np
import warnings
import cv2
import torch
import torch.backends.cudnn as cudnn

import sys

currentUrl = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(currentUrl, 'yolov5')))

cudnn.benchmark = True

class VideoTracker(object):
    def __init__(self, args):
        print('Initialize DeepSORT & YOLO')
        self.args = args

        self.img_size = args.img_size
        self.frame_interval = args.frame_interval
        self.device = select_device(args.device)
        self.half = self.device.type != 'cpu'

        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)
        
        if args.cam != -1:
            print("Using webcam " + str(args.cam))
            self.vdo = cv2.VideoCapture(args.cam)
            if not self.vdo.isOpened():
                print("Cannot open camera")
                exit()
        else:
            self.vdo = cv2.VideoCapture()
        
        cfg = get_config()
        cfg.merge_from_file(args.config_deepsort)

        use_cuda = self.device.type != 'cpu' and torch.cuda.is_available()
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)

        self.detector = torch.load(args.weights, map_location=self.device)['model'].float()
        self.detector.to(self.device).eval()
        if self.half:
            self.detector.half()
        
        self.names = self.detector.module.names if hasattr(self.detector, 'module') else self.detector.names

        print('Initialization done..')
        if self.device == 'cpu':
            warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)
    
    def __enter__(self):
        # ************************ Load video from camera ****************************
        if self.args.cam != -1:
            print('Camera ...')
            ret, frame = self.vdo.read()
            assert ret, "Error: Camera error"
            self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # ************************ Loda video from file *****************************
        else:
            assert os.path.isfile(self.args.input_path), "Path error"
            self.vdo.open(self.args.input_path)
            self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if not self.vdo.isOpened():
                print('Cannot open video from file')
                exit()
            print('Done. Load video file ', self.args.input_path)
        # ************************ Create output ************************************
        if self.args.save_path:
            os.makedirs(self.args.save_path, exist_ok=True)
            self.save_video_path = os.path.join(self.args.save_path, "results.mp4")

            fourcc = cv2.VideoWriter_fourcc(*self.args.fourcc)
            self.writer = cv2.VideoWriter(self.save_video_path, fourcc,
                                          self.vdo.get(cv2.CAP_PROP_FPS), (self.im_width, self.im_height))
            print('Done. Create output file ', self.save_video_path)
        
        if self.args.save_txt:
            os.makedirs(self.args.save_txt, exist_ok=True)

        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.vdo.isOpened():
            self.vdo.release()
        self.writer.release()
        if exc_type:
            print(exc_type, exc_value, exc_traceback)
        
        cv2.destroyAllWindows()
        print('Released video resources.')
    
    def run(self):
        print("Starting running tracker")
        yolo_time, sort_time, avg_fps = [], [], []
        t_start = time.time()

        idx_frame = 0
        last_out = None

        while self.vdo.grab():
            t0 = time.time()
            _, img0 = self.vdo.retrieve()
            # measurement update
            if idx_frame % self.args.frame_interval == 0:
                outputs, yt, st = self.image_track(img0)
                last_out = outputs
                yolo_time.append(yt)
                sort_time.append(st)
                print('Frame %d Done. YOLO-time:(%.3fs) SORT-time:(%.3fs)' % (idx_frame, yt, st))
            # prediction update
            else:
                outputs = last_out
            t1 = time.time()
            avg_fps.append(t1 - t0)

            # post-processing ******************************************      
            # visualize bbox ******************************
            if len(outputs) > 0:
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]
                img0 = draw_boxes(img0, bbox_xyxy, identities)

                text_scale = max(1, img0.shape[1] // 1600)
                cv2.putText(img0, 'frame: %d fps: %.2f ' % (idx_frame, len(avg_fps) / sum(avg_fps)),
                        (20, 20 + text_scale), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=2)
                
            if self.args.save_path:
                self.writer.write(img0)

            if self.args.save_txt:
                with open(self.args.save_txt + str(idx_frame).zfill(4) + '.txt', 'a') as f:
                    for i in range(len(outputs)):
                        x1, y1, x2, y2, idx = outputs[i]
                        f.write('{}\t{}\t{}\t{}\t{}\n'.format(x1, y1, x2, y2, idx))
                
            idx_frame += 1
        
        print('Avg YOLO time (%.3fs), Sort time (%.3fs) per frame' % (sum(yolo_time) / len(yolo_time),
                                                        sum(sort_time) / len(sort_time)))

        t_end = time.time()
        print('Total time (%.3fs), Total Frame: %d' % (t_end - t_start, idx_frame))

    def image_track(self, im0):
        img = letterbox(im0, new_shape=self.img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        s = '%gx%g' % img.shape[2:]

        t1 = time_synchronized()
        with torch.no_grad():
            pred = self.detector(img, augment=self.args.augment)[0]
        pred = non_max_suppression(pred, self.args.conf_thres, self.args.iou_thres,
                                   classes=self.args.classes, agnostic=self.args.agnostic_nms)
        t2 = time_synchronized()

        det = pred[0]

        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            for c in det[:, -1].unique():
                n  = (det[:, -1] == c).sum()
                s += '%g %ss, ' % (n, self.names[int(c)])

            bbox_xywh = xyxy2xywh(det[:, :4]).cpu()
            confs = det[:, 4:5].cpu()

            outputs = self.deepsort.update(bbox_xywh, confs, im0)
        else:
            outputs = torch.zeros((0, 5))
        
        t3 = time.time()
        return outputs, t2 - t1, t3 - t2
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_path', type=str, default='video.mp4', help='source')
    parser.add_argument('--save_path', type=str, default='output/', help='output folder')
    parser.add_argument('--frame_interval', type=int, default=2, help='process every frame interval')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save_txt', default='output/predict/', help='the place where we save the tacks in txt')

    parser.add_argument('--display', action='store_true')
    parser.add_argument('--display_width', type=int, default=800)
    parser.add_argument('--display_height', type=int, default=600)
    parser.add_argument('--camera', action='store', dest='cam', type=int, default='-1')

    parser.add_argument('--weights', type=str, default='yolov5/weights/yolov5s.pt', help='model.pt path')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--classes', nargs='+', type=int, default=[0], help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')

    parser.add_argument('--config_deepsort', type=str, default='./configs/deep_sort.yaml')

    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)
    print(args)

    with VideoTracker(args) as vdo_trk:
        vdo_trk.run()
