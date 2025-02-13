import cv2

from argparse import ArgumentParser

from src.utils import stream_video


CV2_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    "boosting": cv2.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
    "tld": cv2.TrackerTLD_create,
    "medianflow": cv2.TrackerMedianFlow_create,
    "mosse": cv2.TrackerMOSSE_create,
}


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("video_file", type = str)
    parser.add_argument("tracker_type", default = "boosting")

    return parser.parse_args()


def main():
    args = parse_arguments()
    tracker_type = args["tracker_type"].lower()

    tracker = CV2_TRACKERS[tracker_type]()
    init_bbox = None
    
        


if __name__ == "__main__":
    main()