import cv2
import face_recognition
import imutils


class FrameManager:
    
    def __init__(self, path):
        self.__video = cv2.VideoCapture(path)
        if not self.__video.isOpened():
            print("Video nao abriu")
            exit(1)
        self.__num_of_frames = 0

    def grab_frame(self):
        retval, frame = self.__video.read()
        if retval:
            self.__num_of_frames += 1
        else:
            frame = None
        return frame

    def should_analyse(self, frame):
        face_locations = face_recognition.face_locations(frame)
        constain_faces = False
        if len(face_locations) >= 1 :
            constain_faces = True
        return constain_faces
        

    def isRunning(self):
        return self.__video.isOpened()

    def show_image(self, frame):
        cv2.imshow('Video', frame)

    def stop(self):
        self.__video.release()
        cv2.destroyAllWindows()

    def get_num_of_frames(self):
        return str(self.__num_of_frames)
