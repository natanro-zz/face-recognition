#!/usr/bin/env python3

import cv2
import cognitive_face as CF
import os
from time import sleep

import src.actors.avengers as AV

PERSON_GROUP_ID = "avengers_group"

video_path = os.path.abspath(AV.get_video_path())

video = cv2.VideoCapture(video_path)

if not video.isOpened():
    print("Not possible to open video")
    exit(1)

CF.Key.set("XXXXXXXXXXXXXXXXXXXXXXXX")
CF.BaseUrl.set("https://[location].api.cognitive.microsoft.com/face/v1.0/")

# a temporary frame for azure api read it
frame_temp_path = os.path.abspath('out/frame_temp.jpeg')


final_video_img_array = []


def get_rectangle(face_dictionary):
    rec = face_dictionary['faceRectangle']
    left = rec['left']
    top = rec['top']
    bot = left + rec['height']
    right = top + rec['width']
    return (left, top), (bot, right)

def get_name(identified_face):
    candidate_id = identified_face[0]['personId']
    person = CF.person.get("%s" % PERSON_GROUP_ID, candidate_id)
    return person['name']


while True:
    retval, frame = video.read()
    if retval:
        cv2.imwrite(frame_temp_path, frame)
        faces = CF.face.detect(frame_temp_path)
        if len(faces) > 0:
            face_ids = [id['faceId'] for id in faces]
            identified_faces = CF.face.identify(face_ids, PERSON_GROUP_ID)
            for i in range(len(faces)):
                pt1, pt2 = get_rectangle(faces[i])
                cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 3)
                text = 'unknown'
                if len(identified_faces[i]['candidates']) > 0:
                    text = 'Name: '+get_name(identified_faces[i]['candidates']) + ' Confidence:' + str(identified_faces[i]['candidates'][0]['confidence'])
                cv2.putText(frame, text,
                            pt1, cv2.FONT_HERSHEY_DUPLEX, 0.3, (255, 255, 255), 1)
            sleep(3)

        final_video_img_array.append(frame)

        os.remove(frame_temp_path)



    else:
        break

video.release()


h, w, _ = final_video_img_array[0].shape
size = (w, h)

out_video_path = os.path.abspath('out/avengers-azure.avi')
out_video = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'DIVX'), 24, size)

if not out_video.isOpened():
    print("Not possible to open video writer")
    exit(1)

for i in range(len(final_video_img_array)):
    out_video.write(final_video_img_array[i])

out_video.release()
