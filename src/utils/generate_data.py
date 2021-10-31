import numpy as np
import cv2 as cv
import tqdm
import random
import os
import csv

def create_dataset(dataset_folder, img_num):
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    img_floder = os.path.join(dataset_folder, "images")
    if not os.path.exists(img_floder):
        os.makedirs(img_floder)

    csv_file_name = os.path.join(dataset_folder, "label.csv")

    rotateAngle = 0
    startAngle = 0
    endAngle = 360
    thickness = -1
    lineType = 8
    colors = [[0,0,255], [0,255,0], [255,0,0]]

    csv_file_save = []

    for num in tqdm.tqdm(range(img_num)):
        img_save_name = f'{num}.png'

        img = np.ones((256, 256, 3), np.uint8)
        img = img * 255

        # for circle_i in range( random.randint(1, 3) ):

        center_x = random.randint(50, 200) #random.random() * 256
        center_y = random.randint(50, 200) #random.random() * 256
        ptCenter = (center_x, center_y)

        radius = random.randint(30, 80) #random.random() * 256
        ratio =  round( random.random(), 2) / 0.5 + 0.25

        axesSize = (radius, int(radius*ratio))
        label = random.randint(0, 2)
        color = colors[label]
        cv.ellipse(img, ptCenter, axesSize, rotateAngle, startAngle, endAngle, color, thickness, lineType)

        csv_file_save.append([img_save_name, label, center_x, center_y, radius, int(radius*ratio)])

        # cv.line(img, (center_x-radius, center_y-int(radius*ratio)), (center_x+radius, center_y+int(radius*ratio)), (0, 255, 0), thickness=thickness)
        cv.imshow('img', img)
        cv.waitKey()
        cv.imwrite( os.path.join(img_floder, img_save_name), img )

    with open(csv_file_name, mode='w') as csv_file:
        fieldnames = ['img_name', 'label', 'center_x', 'center_y', 'radius_x', 'radius_y']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        for line in csv_file_save:
            writer.writerow({'img_name': line[0],
                             'label': line[1],
                             'center_x': line[2],
                             'center_y': line[3],
                             'radius_x': line[4],
                             'radius_y': line[5]})

if __name__ == '__main__':
    create_dataset('../../data/circle_train', 50)