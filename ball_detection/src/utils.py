import cv2
import matplotlib.pyplot as plt


IMAGE_SIZE = (360, 640, 3)
RED = (0, 0, 255)
BBOX_THICKNESS = 1


def plot_image(image, figax = None, mode = "bgr"):
    if figax is None:
        fig = plt.figure(figsize = (8, 6))
        ax = fig.add_subplot(111)
    else:
        fig, ax = figax

    if mode == "bgr":
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    ax.imshow(image, cmap = "gray" if mode == "gray" else None)
    ax.axis("off")

    return fig, ax


def stream_video(filepath):
    capture = cv2.VideoCapture(filepath)
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_num = 1

    while frame_num < frame_count:
        ret, frame = capture.read()
        if not ret:
            break

        yield frame, frame_num

def show_predictions(yolo, capture):
    while True:
        ret, frame = capture.read()
        if not ret:
            break

        results = yolo(frame).pandas().xyxy[0]
        for _, row in results.iterrows():
            if row["name"] != "sports ball":
                continue

            x1, y1, x2, y2 = row[["xmin", "ymin", "xmax", "ymax"]].astype(int)
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                RED,
                BBOX_THICKNESS,
            )
            
        cv2.imshow("retrained yolo", frame)
        key = cv2.waitKey(30)

        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    return