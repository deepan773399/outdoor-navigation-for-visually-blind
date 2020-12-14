#For object detection

#model is darknet53.conv.74 that we trainned by our COCO dataset
#You can download the model by darknet repo on github

#For distance estimation we developour own model by trainning it with some values
#In this model we need to find a relation between the y coordinate  of the lowest edge of the rectangle (which is plotted on object detected) and the distance that we want to estimate.
#model is-


import tensorflow as tf
from tensorflow import keras

model1 = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model1.compile(optimizer='sgd', loss='mean_squared_error')
xs = np.array([116.0,110.0, 102.0, 92.0, 80.0], dtype=float)
ys = np.array([320.0, 280.0, 240.0, 200.0, 160.0], dtype=float)
model1.fit(xs, ys, epochs=5)

t=model1.predict([y])
