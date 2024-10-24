import cv2
import numpy as np
import logging
import torch
import torchvision.transforms as transforms

from .model import ReNet

class Extractor(object):
    def __init__(self, model_path, use_cuda=True):
        self.net = ReNet(reid=True)
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)['net_dict']
        self.net.load_state_dict(state_dict)
        logger = logging.getLogger("root.tracker")
        logger.info("Loading weights from {}... Done!".format(model_path))
        self.net.to(self.device)
        self.size = (64, 128)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def _preprocess(self, im_crops):
        """
        1. to float with scale from 0 to 1, resize to (64, 128)
        2. to torch tensor and normalization is applied
        3. Add extra dimension from [C, H, W] -> [1, C, H, W]
        4. Concatenate the images along 0th dimension [B, C, H, W]
        5. Make the tensor type float
        """
        def _resize(im, size):
            if im is None:
                raise ValueError("Image is None, cannot resize.")
            im_resized = cv2.resize(im.astype(np.float32)/255., size)
            return im_resized
        
        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops], dim=0).float()
        return im_batch
    
    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.net(im_batch)
        return features.cpu().numpy()

if __name__ == '__main__':
    # img = cv2.imread("demo.jpg")[:, :, (2, 1, 0)]
    # print(img.shape)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # im_crops = [img]
    # extr = Extractor("checkpoint/ckpt.t7")
    # feature = extr(im_crops)
    # print(feature)
    model_path = 'checkpoint/ckpt.t7'
    net = ReNet(reid=True)
    print(net)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)['net_dict']
    # for key in state_dict.keys():
    #     print(key)
    # net.load_state_dict(state_dict)
    logger = logging.getLogger("root.tracker")
    logger.info("Loading weights from {}... Done!".format(model_path))
