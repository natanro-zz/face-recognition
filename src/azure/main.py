from src.frame_manager.FrameManager import FrameManager
from .api import FaceRecognition
import cv2

face_api = FaceRecognition("avengers_group")

video_path = '/home/natrodrigues/Documents/PIBITI/face-recognition/videos/vingadores.mp4'
frame_manager = FrameManager(video_path)

while frame_manager.isRunning():
    frame = frame_manager.grab_frame()
    if frame is not None:
        if face_api.detect_faces(frame): # quem detecta faces é a própria API da Azure
            pass #TODO: API
        frame_manager.show_image(frame) #TODO: trocar show image por video write (análogo)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
frame_manager.stop()
print(frame_manager.get_num_of_frames())

#TODO: encontre os personagens