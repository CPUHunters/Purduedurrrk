import cv2
import os

data_dir = r'C:\Users\SEOLHEE.DESKTOP-VIR1P4P\Downloads\CUB_200_2011\CUB_200_2011\images'
edited_data_dir = r"C:\Users\SEOLHEE.DESKTOP-VIR1P4P\PycharmProjects\image_resizing\data\birds"

IMAGE_SIZE = 72


# crop the image as square to focus on birds
def img_trim_bird(img):
    h, w = img.shape[:2]
    # x, y: 자르고 싶은 지점 왼쪽위 좌표
    trim_img = None
    if w > h:
        x = int((w-h)/2)
        y = 0
        trim_img = img[y:y+h, x:x+h]
    elif w < h:
        x = 0
        y = int((h-w)/2)
        trim_img = img[y:y+w, x:x+w]
    else:
        trim_img = img
    return trim_img


# crop the side part of image to collect the nonbird dataset
def img_trim_nonbird(img):
    h, w = img.shape[:2]
    trim_img = img[50:50+IMAGE_SIZE, w-50-IMAGE_SIZE:w-50]
    return trim_img


def read_img(filepath):
    img = cv2.imread(filepath)
    return img


def main():
    dirnames = os.listdir(data_dir)
    for dirname in dirnames:
        dirpath = os.path.join(data_dir, dirname)
        edited_dirpath = os.path.join(edited_data_dir, dirname)
        filenames = os.listdir(dirpath)
        if not (os.path.exists(edited_dirpath)):
            os.mkdir(edited_dirpath)
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            edited_filepath = os.path.join(edited_dirpath, filename)
            print(edited_filepath)
            origin_img = read_img(filepath)
            # trim_img = img_trim_nonbird(origin_img)
            trim_img = img_trim_bird(origin_img)
            resized_img = cv2.resize(trim_img, (IMAGE_SIZE, IMAGE_SIZE))
            cv2.imwrite(edited_filepath, resized_img)


if __name__ == '__main__':
    main()
