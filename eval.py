import argparse
import time

import cv2
import matplotlib.pyplot as plt

from detectors import *
from utils import *


def run_benchmark(img, detector, n_iters=100):
    """Run benchmark on a detector."""
    times = []
    for _ in range(n_iters):
        start = time.time()
        detector(img)
        times.append(time.time() - start)
    return np.mean(times)


def main(args):
    img = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (1024, 1024))

    sift = SIFTFrontend()
    # silk = SILKFrontend()
    superpoint = SuperPointFrontend(weights_path="modules/superglue/models/weights/superpoint_v1.pth", cuda=False)

    sift_time = run_benchmark(img, sift)
    print("SIFT:\t\t{:.3f} seconds".format(sift_time))
    sift_kp, sift_desc = sift(img)

    # silk_time = run_benchmark(img, silk)
    # silk_kp, silk_desc = silk(img)

    sp_time = run_benchmark(img, superpoint)
    print("SuperPoint:\t{:.3f} seconds".format(sp_time))
    sp_kp, sp_desc = superpoint(img)

    # plot keypoints
    fig, ax = plt.subplots(2, 2)
    ax = ax.flatten()
    ax[0].imshow(cv2.drawKeypoints(img, sift_kp, None))  # type: ignore
    ax[0].set_title("SIFT (detected {} keypoints in {:.3f} seconds)".format(len(sift_kp), sift_time))
    ax[1].imshow(cv2.drawKeypoints(img, superpoint2opencv_keypoints(sp_kp), None))  # type: ignore
    ax[1].set_title("SuperPoint (detected {} keypoints in {:.3f} seconds)".format(len(sp_kp), sp_time))
    # ax[2].imshow(cv2.drawKeypoints(img, silk_kp, None))  # type: ignore
    # ax[2].set_title("SiLK (detected {} keypoints in {:.3f} seconds)".format(len(silk_kp), silk_time))
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/eval.JPG")
    args = parser.parse_args()

    main(args)
