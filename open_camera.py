import cv2
import time
import os

def put_text(img, text, org, scale=0.6, thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)

def main():
    cam_index = 2          # change if needed (0, 1, 2, ...)
    width, height = 640 , 480  
    save_dir = "captures"
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(cam_index, cv2.CAP_ANY)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {cam_index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    cv2.namedWindow("Camera (q=quit, c=capture)", cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok:
            ok, frame = cap.read()
            if not ok:
                print("Frame grab failed.")
                break

        cv2.namedWindow("Camera (q=quit, c=capture)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Camera (q=quit, c=capture)", width, height)

        h, w = frame.shape[:2]
        put_text(frame, "Press 'c' to capture, 'q' to quit", (10, 60))
        put_text(frame, f"Resolution: {w}x{h}", (10, 90))

        cv2.imshow("Camera (q=quit, c=capture)", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            path = os.path.join(save_dir, f"capture.png")
            cv2.imwrite(path, frame)
            print(f"Saved: {path}")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
