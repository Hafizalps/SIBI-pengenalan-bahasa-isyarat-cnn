import cv2
import numpy as np
import display_gestures

def nothing(x):
    pass

image_x, image_y = 64,64

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# Disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from tensorflow.keras.models import load_model
classifier = load_model('Trained_model.h5')

"""print("Tensorflow Version: ", tf.__version__)
print("Eager mode enabled: ", tf.executing_eagerly())
print("GPU available: ", tf.test.is_gpu_available())
tf.get_logger().setLevel('WARNING')
gpus = tf.config.list_physical_devices('GPU')
print(gpus)
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)"""

def predictor():
       import numpy as np
       from tensorflow.keras.preprocessing import image
       test_image = image.load_img('1.png', target_size=(64, 64))
       test_image = image.img_to_array(test_image)
       test_image = np.expand_dims(test_image, axis = 0)
       result = classifier.predict(test_image)

       if h_a == 1:
           if result[0][0] == 1:
                  return '0'
           elif result[0][1] == 1:
                  return '1'
           elif result[0][2] == 1:
                  return '2'
           elif result[0][3] == 1:
                  return '3'
           elif result[0][4] == 1:
                  return '4'
           elif result[0][5] == 1:
                  return '5'
           elif result[0][6] == 1:
                  return '6'
           elif result[0][7] == 1:
                  return '7'
           elif result[0][8] == 1:
                  return '8'
           elif result[0][9] == 1:
                  return '9'
           else:
                  return 'Tidak terdeteksi'

       elif h_a == 0:
           if result[0][10] == 1:
                  return 'A'
           elif result[0][11] == 1:
                  return 'B'
           elif result[0][12] == 1:
                  return 'C'
           elif result[0][13] == 1:
                  return 'D'
           elif result[0][14] == 1:
                  return 'E'
           elif result[0][15] == 1:
                  return 'F'
           elif result[0][16] == 1:
                  return 'G'
           elif result[0][17] == 1:
                  return 'H'
           elif result[0][18] == 1:
                  return 'I'
           elif result[0][19] == 1:
                  return 'J'
           elif result[0][20] == 1:
                  return 'K'
           elif result[0][21] == 1:
                  return 'L'
           elif result[0][22] == 1:
                  return 'M'
           elif result[0][23] == 1:
                  return 'N'
           elif result[0][24] == 1:
                  return 'O'
           elif result[0][25] == 1:
                  return 'P'
           elif result[0][26] == 1:
                  return 'Q'
           elif result[0][27] == 1:
                  return 'R'
           elif result[0][28] == 1:
                  return 'S'
           elif result[0][29] == 1:
                  return 'T'
           elif result[0][30] == 1:
                  return 'U'
           elif result[0][31] == 1:
                  return 'V'
           elif result[0][32] == 1:
                  return 'W'
           elif result[0][33] == 1:
                  return 'X'
           elif result[0][34] == 1:
                  return 'Y'
           elif result[0][35] == 1:
                  return 'Z'
           else :
                  return 'Tidak terdeteksi'

       elif h_a == 2:
           if result[0][36] == 1:
                  return 'Good Job'
           elif result[0][37] == 1:
                  return 'Hello'
           elif result[0][38] == 1:
                  return 'I love you <3'
           elif result[0][39] == 1:
                  return 'Please'
           elif result[0][40] == 1:
                  return 'Sorry'
           else :
                  return 'Tidak terdeteksi'

cam = cv2.VideoCapture(0)

cv2.namedWindow("Trackbars")
cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("H - A - E", "Trackbars", 0, 2, nothing)

cv2.namedWindow("Sign Language Recognition")

img_counter = 0 

img_text = ''
while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame,1)
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")
    h_a = cv2.getTrackbarPos("H - A - E", "Trackbars")


    img = cv2.rectangle(frame, (425,100),(625,300), (0,255,0), thickness=2, lineType=8, shift=0)

    lower_blue = np.array([l_h, l_s, l_v])
    upper_blue = np.array([u_h, u_s, u_v])
    imcrop = img[102:298, 427:623]
    hsv = cv2.cvtColor(imcrop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    cv2.putText(frame, 'Prediksi- ' + str(img_text), (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 0))
    cv2.imshow("Sign Language Recognition", frame)
    cv2.imshow("mask", mask)
    
    #if cv2.waitKey(1) == ord('c'):
        
    img_name = "1.png"
    save_img = cv2.resize(mask, (image_x, image_y))
    cv2.imwrite(img_name, save_img)
    print("{} written!".format(img_name))
    img_text = predictor()
    print(img_text)

    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()