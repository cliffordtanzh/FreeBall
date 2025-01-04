import cv2
import matplotlib.pyplot as plt


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


def stream_video(frames, fps = 30):
    _, _, _, num_frames = frames.shape
    for i in range(num_frames):
        frame = frames[:, :, :, i]
        cv2.imshow("stream", frame)
        cv2.setWindowProperty("stream", cv2.WND_PROP_TOPMOST, 1)

        key = cv2.waitKey(fps)

        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    return