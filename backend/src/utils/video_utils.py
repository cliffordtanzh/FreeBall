import numpy as np
import cv2


def stream_video(frames: np.ndarray, rgb_input: bool = False) -> None:
    try:
        count = 0
        while True:
            if count >= len(frames):
                break

            frame = frames[count]
            if rgb_input:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)

            cv2.imshow("video stream", frame)
            if cv2.waitKey(-1) == ord("q"):
                break

            count += 1
        
    except Exception as e:
        cv2.destroyAllWindows()
        raise e
    
    finally:
        cv2.destroyAllWindows()
        return