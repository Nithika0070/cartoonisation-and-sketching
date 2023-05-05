x=int(input('ENTER 1.Cartoonisation or 2.Sketching'))
if x==1:
    import cv2
    import numpy as np
    from tkinter.filedialog import *

    photo = askopenfilename()
    img = cv2.imread(photo)

    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grey = cv2.medianBlur(grey, 5)
    edges = cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)

    #cartoonize
    color = cv2.bilateralFilter(img, 9, 250, 250)
    cartoon = cv2.bitwise_and(color, color, mask = edges)

    cv2.imshow("Image", img)
    cv2.imshow("Cartoon", cartoon)

    #save
    cv2.imwrite("cartoon.jpg", cartoon)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
elif x==2:
    import cv2 as cv
    import numpy as np

    camera = cv.VideoCapture(0)

    while True:
        ret, frame = camera.read()
        if ret == False:
            break
        # getting width and height of image
        height, width, _ = frame.shape

        # creating copy of image using resize function
        resizedImage = cv.resize(frame, (width, height),
                                 interpolation=cv.INTER_AREA)

        # creating 3X3 kernel, inorder to sharpen the image /frame
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
        # Appling kernel on the frame using filter2D function .
        sharpenImage = cv.filter2D(resizedImage, -1, kernel)

        # converting image into Grayscale image.
        gray = cv.cvtColor(sharpenImage, cv.COLOR_BGR2GRAY)

        # creating inverse of sharpen image .
        inverseImage = 255-gray

        # applying Gussain Blur on the image .
        bluredImage = cv.GaussianBlur(inverseImage, (15, 15), 0, 0)

        # create a pencilSketch using divide function on opencv .
        pencilSketch = cv.divide(gray, 255-bluredImage, scale=256)

        # show the frame on the screen .
        cv.imshow('Sharpen Image', sharpenImage)
        cv.imshow("pencilSketch", pencilSketch)

        cv.imshow("frame", frame)
        key = cv.waitKey(1)
        if key == ord('q'):
            break
    
    cv.destroyAllWindows()
    camera.release()
