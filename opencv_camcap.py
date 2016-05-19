import numpy as np
import cv2

cap = cv2.VideoCapture(0) # Capture video from camera
fps = cap.get(cv2.CAP_PROP_FPS)
# fps = 0
print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
# Get the width and height of frame

print("Frame Size: %dx%d" % (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)

# Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use the lower case
# out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (width, height))

font = cv2.FONT_HERSHEY_SIMPLEX

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        # frame = cv2.flip(frame,0)

        # write the flipped frame
        # out.write(frame)

        cv2.circle(frame, (100,100), 50, (255,0,255))
        cv2.putText(frame, "PADDA?", (200, 200), font, 1, (0,0,0), 6, cv2.LINE_AA)
        cv2.putText(frame, "PADDA?", (200, 200), font, 1, (255,255,255), 4, cv2.LINE_AA)
        cv2.imshow('frame',frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
            break
    else:
        break

# Release everything if job is finished
cap.release()
cv2.destroyAllWindows()
