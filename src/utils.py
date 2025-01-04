import cv2
import matplotlib.pyplot as plt


def plot_image(image, figax = None, bgr = True):
    if figax is None:
        fig = plt.figure(figsize = (8, 6))
        ax = fig.add_subplot(111)
    else:
        fig, ax = figax

    if bgr:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    ax.imshow(image)
    ax.axis("off")

    return fig, ax


def stream_video(frames):
    _, _, _, num_frames = frames.shape
    for i in range(num_frames):
        frame = frames[:, :, :, i]
        cv2.imshow("stream", frame)
        key = cv2.waitKey(1)

        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    return