import cv2


def superpoint2opencv_descriptors(desc):
    return desc.squeeze().cpu().detach().numpy().transpose(1, 0)


def superpoint2opencv_keypoints(pts):
    return [cv2.KeyPoint(k[0], k[1], 1) for k in pts.squeeze().cpu().detach().numpy()]
