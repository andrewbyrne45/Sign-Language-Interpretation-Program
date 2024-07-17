# program to detect symbols within a video

# imports
from modules import *

# variables for detection
cap = cv.VideoCapture(0)
detector = HandDetector(maxHands=2)

# variables
equalize = 20
size = 300
folder = "Data/Help"
counter = 0

# if label is 0, it will be A. And if label is 1, it will be B etc..
# maybe try to import 'labels.txt' as an easier means, rather than
# manually entering labels
#labels = ["A", "B", "C"]

while True:
    onComp, img = cap.read()
    # secondary image needed for hiding hand points on main screen
    output = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        # detecting one hand and creating a broadcasting box
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # create an np matrix
        crop = img[y - equalize: y + h + equalize, x - equalize: x + w + equalize]
        imgWhite = np.ones((size, size, 3), np.uint8)*255

        # inserting imgCrop into imgWhite
        imgCropShape = crop.shape

        # ensureing the proportion of both height and width
        aspectRatio = h/w
        # if aspectRatio is greater than 1, then height is greater
        if aspectRatio >1:
            cons = size/h
            # rounding up the decimal
            widthCalc = math.ceil(cons * w)
            # resizing the image box to the correct value so that
            # it is always placed in the centre (height)
            imgResize = cv.resize(crop, (widthCalc, size))
            resizeShape = imgResize.shape
            widthGap = math.ceil((size - widthCalc) / 2)
            imgWhite[:, widthGap : widthCalc + widthGap] = imgResize
            # gathering prediction and printing confidence values

        else: 
            cons = size/w
            # rounding up the decimal
            heightCalc = math.ceil(cons * w)
            # resizing the image box to the correct value so that
            # it is always placed in the centre (width)
            imgResize = cv.resize(crop, (size, heightCalc))
            resizeShape = imgResize.shape
            heightGap = math.ceil((size - heightCalc) / 2)
            imgWhite[heightGap : heightCalc + heightGap, :] = imgResize
            # gathering prediction

        cv.rectangle(output, (x - equalize, y - equalize - 50), (x - equalize + 90, y - equalize - 50 + 50), (255, 0, 255), cv.FILLED)
        cv.rectangle(output, (x - equalize, y - equalize), (x + w + equalize, y + h + equalize), (255, 0, 255), 4)
        cv.imshow("ImageCrop", crop)
        cv.imshow("ImageWhite", imgWhite)

        key = cv.waitKey(1)
        if key == ord("s"):
            time.sleep(3)
            counter =+ 1
            cv.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
            print(counter)

    cv.imshow("image", output)
    cv.waitKey(1)