"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
from ex1_utils import *

title_window = 'gamma correction'



def gammaDisplay(img_path: str, rep: int):
    def changeValue(val):
        if rep == 2:
            img = cv2.imread(img_path)
        else:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        img = img.dot(NORMALIZE)
        img = np.power(img, val * 0.01)
        img = cv2.imshow(title_window, img)

    if rep ==2:
        img = cv2.imread(img_path)
    else:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    cv2.imshow(title_window, img)
    cv2.createTrackbar("GAMMA", title_window, 0, 200, changeValue)
    cv2.waitKey(0)
def main():
    gammaDisplay('beach.jpg', LOAD_GRAY_SCALE)


if __name__ == '__main__':
    main()
