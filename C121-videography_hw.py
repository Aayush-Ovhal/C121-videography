import cv2
import numpy as np
import time

fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_file = cv2.VideoWriter("outputfile.mp4", fourcc, 10.0, (1080, 1920))

capture = cv2.VideoCapture(0)

time.sleep(2)
bg = 0

for i in range(60): 
   ret, bg = capture.read()

bg = np.flip(bg, axis = 1)

while(capture.isOpened()): 
    ret, img = capture.read()
    if not ret: 
       break

    img = np.flip(img, axis = 1)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([148, 189, 242])
    upper_blue = np.array([107, 165, 242])

    mask1 = cv2.inRange(hsv, lower_blue, upper_blue)

    lower_blue = np.array([36, 114, 217])
    upper_blue = np.array([13, 101, 217])

    mask2 = cv2.inRange(hsv, lower_blue, upper_blue)

    mask1 = mask1 + mask2

    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE, np.ones((3,3), np.uint8))

    mask2 = cv2.bitwise_not(mask1)

    res1 = cv2.bitwise_and(img, img, mask = mask2)
    res2 = cv2.bitwise_and(bg, bg, mask = mask1)

    final_output = cv2.addWeighted(res1, 1, res2, 1, 0)
    output_file.write(final_output)
    cv2.imshow("output", final_output)
    cv2.waitKey(1)

capture.release()
out.release()
cv2.destroyAllWindows()
