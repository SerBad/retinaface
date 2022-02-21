from retinaface import RetinaFace
import matplotlib.pyplot as plt
import cv2
import time
import os
import shutil
import argparse
import pandas
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import numpy as np


# https://github.com/serengil/retinaface

def detect_faces(img_path: str):
    base_name = os.path.basename(img_path)
    try:
        resp = RetinaFace.detect_faces(img_path, threshold=0.1)
        if len(resp) > 0:
            print(img_path, " - ", resp, )

            last_score = 0
            last_face = []
            for key in resp:
                identity = resp[key]

                confidence = identity["score"]
                # print(img_path, confidence, )
                # left,top,right,bottom
                if confidence > last_score:
                    last_score = confidence
                    last_face = identity["facial_area"]

            print(img_path, last_face)
            return 1, [base_name, [last_face[1], last_face[2], last_face[3], last_face[0]]]

    except Exception as e:
        print(img_path, e)

    shutil.copyfile(img_path, os.path.join(noneFaceRoot, base_name))
    return 0, [base_name, []]


class ImageOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--path', type=str, required=True, default='results', help='image file path')
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        return self.opt


if __name__ == '__main__':
    opt = ImageOptions().parse()
    print('start classification image', opt)
    tt = time.time()
    path = opt.path
    flist = os.listdir(path)
    noneFaceRoot = path + "_NoneFaceRoot"
    if not os.path.exists(noneFaceRoot):
        os.makedirs(noneFaceRoot)

    executor = ProcessPoolExecutor(max_workers=1)
    columns1 = ["image", "face[top, right, bottom, left]"]
    resultFaceData = []
    for index1 in range(0, len(flist)):
        image = path + os.sep + flist[index1]
        futures = []
        task = executor.submit(detect_faces, image)
        futures.append(task)

        for future in as_completed(futures):
            is_face, data = future.result()
            resultFaceData.append(data)

            futures.remove(future)
            print('progress============:', len(resultFaceData), "/", len(flist))

            if len(resultFaceData) == len(flist):
                resultFace = pandas.ExcelWriter(path + os.sep + "tags.xlsx")

                pandas.DataFrame(resultFaceData, columns=columns1).to_excel(resultFace, index=False)
                resultFace.save()

    print('time2 end:', time.time() - tt)
