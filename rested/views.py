import cv2
import numpy as np
import pandas as pd
from .apps import RestedConfig
from rest_framework.views import APIView
from rest_framework.response import Response


class WeightPrediction(APIView):
    """
    We use Weight Prediction for example implement APIs using Django REST Framework
    """
    def post(self, request):
        """
        Usage: {"Height": 172, "Gender: "Female"}
        :param request:
        :return:
        """
        data = request.data
        height = data['Height']
        gender = data['Gender']

        if gender == 'Male':
            gender = 0

        elif gender == 'Female':
            gender = 1

        # Python 3.10 or Above
        # match gender:
        #     case 'Male':
        #         gender = 0
        #
        #     case 'Female':
        #         gender = 1

        else:
            return Response("Gender field is invalid", status=400)

        model_linear_regression = RestedConfig.model
        weight_predicted = model_linear_regression.predict([[gender, height]])[0][0]
        weight_predicted = np.round(weight_predicted, 1)
        response_dict = {"Predicted Weight (kg)": weight_predicted}

        return Response(response_dict, status=200)


# class VideoEmotion(APIView):
#     def extract_face_features(self, faces, offset_coefficients=(0.075, 0.05)):
#         gray = faces[0]
#         detected_face = faces[1]
#         new_face = []
#
#         for det in detected_face:
#             x, y, w, h = det
#             horizont_offset = np.int(np.floor(offset_coefficients[0] * w))
#             vertica_offset = np.int(np.floor(offset_coefficients[1] * h))
#             extracted_face = gray[y + vertica_offset:y + h,
#                              x + horizont_offset:x - horizont_offset]
#             new_extracted_face = zoom(extracted_face,
#                                       48 / extracted_face.shape[0],
#                                       48 / extracted_face.shape[1])
#             new_extracted_face = new_extracted_face.astype(np.float32)
#             new_extracted_face /= float(new_extracted_face.max())
#             new_face.append(new_extracted_face)
#
#         return new_face
#
#     @staticmethod
#     def detect_face(self, frame):
#         cascade_path = "uploads/models/cascade_frontal_face_default.xml"
#         face_cascade = cv2.CascadeClassifier(cascade_path)
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         detected_faces = face_cascade.detectMultiScale(gray,
#                                                        scaleFactor=1.1,
#                                                        minNeighbors=,
#                                                        minSize=(48, 48),
#                                                        flags=cv2.CASCADE_SCALE_IMAGE)
#         coord = []
#
#         for x, y, w, h in detected_faces:
#             if w > 100:
#                 sub_ing = frame[y:y + h, x:x + w]
#                 coord.append([x, y, w, h])
#
#         return gray, detected_faces, coord
#
#     def get_frame(self, vid_cap, sec, count):
#         vid_cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
#         has_frames, image = vid_cap.read()
#
#         if has_frames:
#             cv.inwrite("frames/frame" + str(count) + ".")
#
#         return has_frames
#
#     def emotion_extract(self, image_gen):
#         for face in VideoEmotion.extract_face_features(VideoEmotion.detect_face(image_gen)):
#             to_predict = np.reshape(face.flatten(), (1, 48, 48, 1))
#             res_prep = CoreConfig.loaded_model.predict(to_predict)
#             result_num = np.argmax(res_prep)
#
#             return result_num
#
#     def freq_persona(self, my_list):
#         count_add = 0
#         freq = {}
#         a1, a2 = [], []
#
#         for item in my_list:
#             if item in freq:
#                 freq[item] += 1
#
#             else:
#                 freq[item] = 1
#
#         for emo_key, emo_count in freq.items():
#             a1.append(emo_key)
#             a2.append(emo_count)
#
#         d_tog = {'Emotion': a1, 'Count': a2}
#         df_emo = pd.DataFrame(d_tog)
#
#         return df_emo
