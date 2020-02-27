import cv2

videoFile = 'sample_videos/multipleTags.mp4'
cap = cv2.VideoCapture(videoFile)
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        cv2.imshow('frame', cv2.resize(frame, (720, 480)))
    else:
        break

print('here')
cap.release()