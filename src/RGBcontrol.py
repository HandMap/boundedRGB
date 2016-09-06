import numpy as np
import cv2

cap = cv2.VideoCapture(0)
ret, image = cap.read()


def nothing(*arg):
    pass


def onMouse(event, x, y, flags, param):
    # grab references to the global variables

    if event == cv2.EVENT_MOUSEMOVE:
        refPt = [(x, y)]
        (blue, green, red) = event[refPt[0]]
        cv2.setTrackbarPos('mouseOver - red', 'RGB', red)
        cv2.setTrackbarPos('mouseOver - green', 'RGB', green)
        cv2.setTrackbarPos('mouseOver - blue', 'RGB', blue)


cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', onMouse)
cv2.createTrackbar('lower - red', 'RGB', 0, 255, nothing)
cv2.createTrackbar('lower - green', 'RGB', 0, 255, nothing)
cv2.createTrackbar('lower - blue', 'RGB', 0, 255, nothing)

cv2.createTrackbar('upper - red', 'RGB', 1, 255, nothing)
cv2.createTrackbar('upper - green', 'RGB', 1, 255, nothing)
cv2.createTrackbar('upper - blue', 'RGB', 1, 255, nothing)

cv2.createTrackbar('mouseOver - red', 'RGB', 0, 255, nothing)
cv2.createTrackbar('mouseOver - green', 'RGB', 0, 255, nothing)
cv2.createTrackbar('mouseOver - blue', 'RGB', 0, 255, nothing)

while True:
    # Capture frame-by-frame
    ret, image = cap.read()

    thrs1 = cv2.getTrackbarPos('lower - red', 'RGB')
    thrs2 = cv2.getTrackbarPos('lower - green', 'RGB')
    thrs3 = cv2.getTrackbarPos('lower - blue', 'RGB')
    thrs4 = cv2.getTrackbarPos('upper - red', 'RGB')
    thrs5 = cv2.getTrackbarPos('upper - green', 'RGB')
    thrs6 = cv2.getTrackbarPos('upper - blue', 'RGB')
    thrs7 = cv2.getTrackbarPos('mouseOver - red', 'RGB')
    thrs8 = cv2.getTrackbarPos('mouseOver - green', 'RGB')
    thrs9 = cv2.getTrackbarPos('mouseOver - blue', 'RGB')

    if (thrs1 > thrs4):
        cv2.setTrackbarPos('lower - red', 'RGB', thrs4 - 1)
    if (thrs2 > thrs5):
        cv2.setTrackbarPos('lower - green', 'RGB', thrs5 - 1)
    if (thrs3 > thrs6):
        cv2.setTrackbarPos('lower - blue', 'RGB', thrs6 - 1)

    # define the list of boundaries
    boundaries = [
        ([thrs3, thrs2, thrs1], [thrs6, thrs5, thrs4])
    ]

    # loop over the boundaries
    for (lower, upper) in boundaries:
        # create NumPy arrays from the boundaries
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")

        # find the colors within the specified boundaries and apply
        # the mask
        mask = cv2.inRange(image, lower, upper)
        output = cv2.bitwise_and(image, image, mask=mask)

        imageOut = np.hstack([image, output])

    # Display the resulting frame
    cv2.imshow('RGB', imageOut)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
