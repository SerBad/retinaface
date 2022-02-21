from retinaface import RetinaFace
import matplotlib.pyplot as plt
import cv2
import os

path = "./dataset"

flist = os.listdir(path)
for index1 in range(0, len(flist)):

    img_path = path + os.sep + flist[index1]
    base_name = os.path.basename(img_path)

    img = cv2.imread(img_path)

    resp = RetinaFace.detect_faces(img_path, threshold=0.1)

    for key in resp:
        # print(img_path, key)
        identity = resp[key]
        print(identity)

        confidence = identity["score"]
        rectangle_color = (255, 0, 0)
        score = identity["score"]
        if score > 0.6:
            # left,top,right,bottom

            facial_area = identity["facial_area"]
            print(img_path, facial_area)

            cv2.rectangle(img, (facial_area[2], facial_area[3]), (facial_area[0], facial_area[1]), rectangle_color, 10)

    plt.imshow(img[:, :, ::-1])
    plt.axis('off')
    plt.show()
    cv2.imwrite('./outputs' + os.sep + base_name, img)
