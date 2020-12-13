#For model development we use darknet and train our model by using labelled data after do labling on the data
#after that a weight file is generated
#and by using that weight file and yolo.cfg file we detect the object in real by the object_detect_real_time.py file

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


net = cv2.dnn.readNetFromDarknet("C:/Users/deepa/Desktop/yolo/yolov2.cfg", r"C:/Users/deepa/Desktop/yolo/yolov2.weights")
classes = []
with open('C:/Users/deepa/Desktop/yolo/coco.names','r') as f:
    classes=[line.strip() for line in f.readlines()]


#for distance estimation
model1 = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model1.compile(optimizer='sgd', loss='mean_squared_error')
xs = np.array([116.0,110.0, 102.0, 92.0, 80.0], dtype=float)
ys = np.array([320.0, 280.0, 240.0, 200.0, 160.0], dtype=float)
model1.fit(xs, ys, epochs=5)

cap = cv2.VideoCapture(0)

while 1:
    _, img = cap.read()
    img = cv2.resize(img, (600, 600))
    hight, width, _ = img.shape
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)

    net.setInput(blob)

    output_layers_name = net.getUnconnectedOutLayersNames()

    layerOutputs = net.forward(output_layers_name)

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            score = detection[5:]
            class_id = np.argmax(score)
            confidence = score[class_id]
            if confidence > 0.7:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * hight)
                w = int(detection[2] * width)
                h = int(detection[3] * hight)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, .5, .4)

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            score = detection[5:]
            class_id = np.argmax(score)
            confidence = score[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * hight)
                w = int(detection[2] * width)
                h = int(detection[3] * hight)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, .8, .4)
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(boxes), 3))
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = colors[i]
            print(y)
            #print(model.predict([10.0]))
            t=model1.predict([y])
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label + " " + confidence, (x, y + 400), font, 2, color, 2)
            cv2.putText(img,"distance - " +str(t), (100, 100), font, 2, color, 2)

    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
