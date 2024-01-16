from typing import Any
import cv2
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor

# from disk import DISK
# from modules.silk.scripts.examples.common import get_model
# from silk.backbones.silk.silk import from_feature_coords_to_image_coords
from superglue.superpoint import SuperPoint


# class DISKFrontend:
#    def __init__(self, weights_path, cuda=False):
#        self.cuda = cuda
#        self.detector = DISK()
#        self.detector.load_state_dict(torch.load(weights_path, map_location="cpu"))
#        if self.cuda:
#            self.detector = self.detector.cuda()
#        self.detector.eval()
#
#    def __call__(self, img):
#        pass


class SIFTFrontend:
    def __init__(self):
        self.detector = cv2.SIFT_create()  # type: ignore

    def __call__(self, img):
        return self.detector.detectAndCompute(img, None)


# class SILKFrontend:
#    def __init(self):
#        self.detector = get_model(default_outputs=("sparse_positions", "sparse_descriptors"))
#
#    def __call__(self, img):
#        kps, descs = self.detector(img)
#        return from_feature_coords_to_image_coords(self.detector, kps), descs


class SuperPointFrontend(object):
    """Wrapper around pytorch net to help with pre and post image processing."""

    def __init__(self, weights_path, cuda=False):
        self.cuda = cuda
        self.detector = SuperPoint({"weights_path": weights_path})
        if self.cuda:
            self.detector = self.detector.cuda()
        self.detector.eval()

    def __call__(self, img):
        if self.cuda:
            input = {"image": to_tensor(img.astype("float32") / 255.0).unsqueeze(0).cuda()}
        else:
            input = {"image": to_tensor(img.astype("float32") / 255.0).unsqueeze(0)}
        output = self.detector(input)
        return output["keypoints"][0], output["descriptors"][0]
