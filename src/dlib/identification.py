import os
import pickle

import cv2
import face_recognition
from PIL import Image

import src.actors.avengers as AV

FRAME_TEMP_JPEG = os.path.abspath('frame_temp.jpeg')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.6):
    """
    Recognizes faces in given image using a trained KNN classifier
    :param X_img_path: path to image to be recognized
    :param knn_clf: (optional) a knn classifier object. if not specified, model_save_path must be specified.
    :param model_path: (optional) path to a pickled knn classifier. if not specified, model_save_path must be knn_clf.
    :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance
           of mis-classifying an unknown person as a known one.
    :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
        For faces of unrecognized persons, the name 'unknown' will be returned.
    """
    if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception("Invalid image path: {}".format(X_img_path))

    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Load image file and find face locations
    X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(X_img)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in
            zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]


def frames_processed(predictions):
    frame = cv2.imread(FRAME_TEMP_JPEG)

    for name, (top, right, bottom, left) in predictions:
        cv2.rectangle(frame, (bottom, left), (right, top), (0, 255, 0), 3)
        cv2.putText(frame, name, (left, top), cv2.FONT_HERSHEY_DUPLEX,
                    0.3, (255, 255, 255), 1)

    return frame

if __name__ == '__main__':
    video_path_in = AV.get_video_path()
    video = cv2.VideoCapture(video_path_in)

    if not video.isOpened():
        print("Nao foi possivel abrir video_in")
        exit(1)

    frames_array = []
    while True:
        retval, frame = video.read()
        if retval:
            rgb_frame = frame[:, :, ::-1]
            rgb_frame = Image.fromarray(rgb_frame)
            rgb_frame.save(FRAME_TEMP_JPEG)

            predictions = predict(FRAME_TEMP_JPEG, model_path='trained_knn_model.clf')

            frame_to_be_recorded = frames_processed(predictions)
            frames_array.append(frame_to_be_recorded)

            os.remove(FRAME_TEMP_JPEG)

        else:
            break


    h, w, _ = frames_array[0].shape
    size = (w,h)
    out_video_path = os.path.abspath('out/vingadores-dlib.avi')
    out_video = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'DIVX'), 24, size)

    for i in range(len(frames_array)):
        out_video.write(frames_array[i])

    out_video.release()
