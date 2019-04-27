import cv2
import os
import face_recognition
import shutil

imagePath = "C:/Users/chuan/OneDrive/Desktop/Test/Test/Elon.jpg"
cascPath = "C:/Users/chuan/PycharmProjects/FaceDetection/haarcascade_frontalface_default.xml"

loadPath = "C:/Users/chuan/OneDrive/Desktop/Test/Load/"

load_image_array = []
def load_image():
    for filename in os.listdir(loadPath):
        load_image_array.append(filename)
load_image()

directory = "C:/Users/chuan/OneDrive/Desktop/Test/"

known_face_names = []
known_face_encodings = []


def addImage_Encode(name):
    _image = face_recognition.load_image_file("C:/Users/chuan/OneDrive/Desktop/Test/" + name + ".jpg")
    print(_image)
    face_encoding = face_recognition.face_encodings(_image)[0]
    known_face_names.append(name)
    known_face_encodings.append(face_encoding)


def test_new_Face(test_face_encode):
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            _image = face_recognition.load_image_file("C:/Users/chuan/OneDrive/Desktop/Test/" + filename)
            face_encoding = face_recognition.face_encodings(_image)[0]
            statement = face_recognition.compare_faces([face_encoding], test_face_encode, tolerance=0.3)
            print(filename, statement)
            if statement[0]:
                print("face found!")
                return True;

    return False;



def findface(imagePath):
    faceCascade = cv2.CascadeClassifier(cascPath)
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=6,
        minSize=(100, 100),
        flags=cv2.FONT_HERSHEY_SIMPLEX
    )

    print("Found {0} faces!".format(len(faces)))
    print(load_image_array)

    # Draw a rectangle around the faces
    img_counter = 0
    for (x, y, w, h) in faces:

        img = cv2.imread(imagePath)
        crop_img = img[y:y + h, x:x + w]
        # cv2.imshow("cropped", crop_img)
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, crop_img)

        test_image_encode = face_recognition.load_image_file(img_name)
        test_face_encode = face_recognition.face_encodings(test_image_encode)[0]

        if not test_new_Face(test_face_encode):
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow("Unknown Face", crop_img)
            cv2.waitKey(0)
            print("press esc to close frame")
            new_file_name = input("input name: ")
            os.remove(img_name)
            cv2.imwrite(new_file_name + ".jpg", crop_img)
            os.rename(new_file_name + ".jpg", directory + new_file_name + ".jpg")


        else:
            cv2.rectangle(image, (x, y), (x + w, y + h), (202, 20, 202), 2)
            os.remove(img_name)
        print("{} written!".format(img_name))
        img_counter += 1

    cv2.waitKey(0)

load_image_pivit = 0
print(load_image_array)
for filename in load_image_array:
    path = loadPath + load_image_array[load_image_pivit]
    print(path)
    findface(path)
    load_image_pivit+=1

video_capture = cv2.VideoCapture(0)

known_face_names = []
known_face_encodings = []


def addImage_Encode(name):
    _image = face_recognition.load_image_file("C:/Users/chuan/OneDrive/Desktop/Test/" + name + ".jpg")
    face_encoding = face_recognition.face_encodings(_image)[0]
    known_face_names.append(name)
    known_face_encodings.append(face_encoding)


for filename in os.listdir(directory):
    if filename.endswith(".jpg"):
        print(filename)
        addImage_Encode(filename.replace(".jpg",""))


face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

name = " "
while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        if name == "Unknown":
            cv2.rectangle(frame, (left, top), (right, bottom), (50, 0, 50), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        else:
        # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (50, 202, 50), 2)

        # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
'''
import cv2

cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter = 0

while True:
    ret, frame = cam.read()
    cv2.imshow("test", frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()
'''