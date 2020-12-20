import time
import cv2

class Camera:
    def cameraMod():
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50,50)
        fontScale = 1
        color = (255, 0,0)
        thickness = 2

        new_frame_time = 0
        prev_frame_time = 0

        vid = cv2.VideoCapture(0)
        ret, frame = vid.read()
        while (ret):
            ret, frame = vid.read()
            new_frame_time = time.time()
            fps = 1/(new_frame_time-prev_frame_time)
            prev_frame_time = new_frame_time

            fps = str(int(fps))

            cv2.putText(frame, 'FPS: '+fps, org, font, fontScale, color, thickness, cv2.LINE_AA)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        vid.release()
        cv2.destroyAllWindows()
if __name__ == '__main__':
    Camera.cameraMod()

