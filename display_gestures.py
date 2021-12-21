import cv2
import numpy as np

gestures = []
jumlahGesture = 42
blankImage = np.zeros((64,64), np.uint8)

def stackImage(scale, images):
    width = images[0][0].shape[1]
    height = images[0][0].shape[0]
    ver = None
    for img in images:
        hor = None
        for i in img:
            if i.shape[:2] == images[0][0].shape[:2]:
                i = cv2.resize(i, (0, 0), None, scale, scale)
            else:
                i = cv2.resize(i, (width, height), None, scale, scale)

            if len(i.shape) == 2:
                i = cv2.cvtColor(i, cv2.COLOR_GRAY2BGR)

            if hor is not None:
                hor = np.hstack((hor, i))
            else:
                hor = i
        if ver is not None:
            ver = np.vstack((ver, hor))
        else:
            ver = hor
    return ver

for j in range(jumlahGesture):
	gestures.append(cv2.imread('gestures/' + str(j+1) + '.png', 0))

kumpulanGestur = stackImage(1, ([gestures[0], gestures[1], gestures[2], gestures[3], gestures[4]],
								   [gestures[5], gestures[6], gestures[7], gestures[8], gestures[9]],
								   [gestures[10], gestures[11], gestures[12], gestures[13], gestures[14]],
								   [gestures[15], gestures[16], gestures[17], gestures[18], gestures[19]],
								   [gestures[20], gestures[21], gestures[22], gestures[23], gestures[24]],
								   [gestures[25], blankImage, blankImage, blankImage, blankImage],
                                   [gestures[27], gestures[28], gestures[29], gestures[30], gestures[31]],
                                   [gestures[32], gestures[33], gestures[34], gestures[35], gestures[36]],
                                   [gestures[37], gestures[38], gestures[39], gestures[40], gestures[41]]))

cv2.imshow('Kamus SIBI', kumpulanGestur)

#cv2.waitKey(0)
#cv2.destroyAllWindows()