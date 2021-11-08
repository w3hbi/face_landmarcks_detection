# import required packages :
import cv2
import mediapipe as mp
import time


# this function is to confirm that you want to use camera or not :
def start():
    print('Welcome to 3D face detection!\n')
    while True:
        print('Do you want to proceed (Y/N): \n')
        choice = str(input('Choice : '))

        if choice in ['Y', 'N', 'y', 'n']:
            return choice
        break


# Create camera properties and and initialize Drawing tools
def create_capture():
    MP_FACE_MESH = mp.solutions.face_mesh
    FACE_MESH = MP_FACE_MESH.FaceMesh()
    MP_DRAW = mp.solutions.drawing_utils

    CAPTURE = cv2.VideoCapture(0)

    return MP_FACE_MESH, FACE_MESH, MP_DRAW, CAPTURE


# Main Function :
if __name__ == '__main__':

    pTime = 0

    res = start()
    if res == 'Y' or res == 'y':
        (mp_face_mesh, face_mesh, mp_draw, capture) = create_capture()
        while True:
            ret, img = capture.read()

            results = face_mesh.process(img)

            if results.multi_face_landmarks:
                for landmarks in results.multi_face_landmarks:
                    mp_draw.draw_landmarks(
                        img,
                        landmarks,
                        mp_face_mesh.FACEMESH_TESSELATION,
                        mp_draw.DrawingSpec((8, 0, 255), 1, 1),
                        mp_draw.DrawingSpec((255, 0, 8), 1, 1)
                    )
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime

            cv2.putText(img, f'FPS:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('386 Point Face Landmarks Detection', img)
            cv2.waitKey(20)

            if cv2.getWindowProperty('386 Point Face Landmarks Detection', cv2.WND_PROP_VISIBLE) < 1:
                print('Bye Bye !')
                break

        cv2.destroyAllWindows()

    else:
        print('Bye Bye !')
