import numpy as np
import cv2
from openalpr import Alpr
import sys

RTSP_SOURCE  = "rtsp://192.168.1.150:8554/webCamStream"
WINDOW_NAME  = 'openalpr'
FRAME_SKIP   = 15


def open_cam_rtsp(uri, width=1280, height=720, latency=2000):
    gst_str = ('rtspsrc location={} ! appsink').format(uri)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


def main():
    alpr = Alpr('eu', 'openalpr.conf', '/usr/share/openalpr/runtime_data')
    if not alpr.is_loaded():
        print('Error loading OpenALPR')
        sys.exit(1)
    alpr.set_top_n(1)

    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("traffic.mp4")
    if not cap.isOpened():
        alpr.unload()
        sys.exit('Failed to open video file!')
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setWindowTitle(WINDOW_NAME, 'OpenALPR video test')

    _frame_number = 0
    while True:
        ret_val, frame = cap.read()
        if not ret_val:
            print('VideoCapture.read() failed. Exiting...')
            break

        _frame_number += 1
        if _frame_number % FRAME_SKIP != 0:
            continue

        results = alpr.recognize_ndarray(frame)
        for plate in results['results']:
            box = plate['coordinates']
            if box:
                # Ensure the bounding box coordinates are valid
                pts = np.array([[point['x'], point['y']] for point in box], np.int32)
                if pts.shape[0] > 0:
                    pts = pts.reshape((-1, 1, 2))
                    # Draw bounding box on the frame
                    cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

            # Extract and print plate information
            best_candidate = plate['candidates'][0]
            print('Plate: {:7s} ({:.2f}%)'.format(best_candidate['plate'].upper(), best_candidate['confidence']))

        # Display the frame with bounding boxes
        cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
    cap.release()
    alpr.unload()


if __name__ == "__main__":
    main()
