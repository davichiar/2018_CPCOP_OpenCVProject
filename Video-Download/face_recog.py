# face_recog.py

import face_recognition
import cv2
import camera
import os
import numpy as np

class FaceRecog():
    def __init__(self):
        # OpenCV를 사용하여 장치 0에서 캡처. 캡처에 문제가 있는 경우
        # 웹캠에서 아래 줄을 읽고 비디오 파일을 사용하십시오.
        self.camera = camera.VideoCamera()

        self.known_face_encodings = []
        self.known_face_names = []

        # 사진을 업로드 하여 인지하는 것 확인
        dirname = 'knowns'
        files = os.listdir(dirname)
        for filename in files:
            name, ext = os.path.splitext(filename)
            if ext == '.jpg':
                self.known_face_names.append(name)
                pathname = os.path.join(dirname, filename)
                img = face_recognition.load_image_file(pathname)
                face_encoding = face_recognition.face_encodings(img)[0]
                self.known_face_encodings.append(face_encoding)

        # 변수 초기화
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.process_this_frame = True

    def __del__(self):
        del self.camera

    def get_frame(self):
        # 비디오 프레임 수집
        frame = self.camera.get_frame()

        # 더 빠른 얼굴 인식 처리를 위해 비디오 프레임의 크기를 1/4로 조정
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # 이미지를 BGR 색상(OpenCV에서 사용하는 색상)에서 RGB 색상(Face_Inception에서 사용하는 색상)으로 변환
        rgb_small_frame = small_frame[:, :, ::-1]

        # 다른 모든 비디오 프레임만 처리하여 시간 절약
        if self.process_this_frame:
            # 현재 비디오 프레임에서 모든 얼굴과 얼굴 인코딩 찾기
            self.face_locations = face_recognition.face_locations(rgb_small_frame)
            self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

            self.face_names = []
            for face_encoding in self.face_encodings:
                # 얼굴이 알려진 얼굴과 일치하는지 확인
                distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                min_value = min(distances)

                # 공차: 일치한다고 간주할 수 있는 얼굴 사이의 거리. 낮은 쪽이 더 엄격하다.
                # 0.6은 전형적인 최고의 공연이다.
                name = "Unknown"
                if min_value < 0.6:
                    index = np.argmin(distances)
                    name = self.known_face_names[index]

                self.face_names.append(name)

        self.process_this_frame = not self.process_this_frame

        # 결과 표시
        for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
            # 감지한 프레임이 1/4 크기로 확장되었기 때문에 배면 위치 스케일업
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # 얼굴 주위에 상자 그리기
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # 얼굴 아래에 이름표를 그려라.
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        return frame

    def get_jpg_bytes(self):
        frame = self.get_frame()
        # 모션 JPEG를 사용하고 있지만, OpenCV는 원시 이미지를 캡처하기 위해 기본 설정,
        # 따라서 JPEG로 인코딩하여
        # 비디오 스트림.
        ret, jpg = cv2.imencode('.jpg', frame)
        return jpg.tobytes()


if __name__ == '__main__':
    face_recog = FaceRecog()
    print(face_recog.known_face_names)
    while True:
        frame = face_recog.get_frame()

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    print('finish')
