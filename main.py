import tensorflow as tf
import numpy as np
from PIL import Image
import os

mnist = tf.keras.datasets.mnist
(xtrain, ytrain), (xtest, ytest) = mnist.load_data()
xtrain = tf.keras.utils.normalize(xtrain, axis=1)
xtest = tf.keras.utils.normalize(xtest, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dense(10, activation="softmax"))
model.compile(optimizer="Adamax", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(xtrain, ytrain, epochs=3)
model.save("first.model")
currentmodel = tf.keras.models.load_model("first.model")

current = 1
while os.path.isfile(f"numberpictures/number{current}.png"):
    img = Image.open(f"numberpictures/number{current}.png").convert("L")
    img = img.resize((28, 28))
    img_array = np.invert(np.array(img))
    img_array = img_array.reshape(1, 28, 28)
    img_array = img_array / 255.0
    # for x in range(28):
    #     for y in range(28):
    #         if img_array[0, x, y] > 0:
    #             img_array[0, x, y] = 1
    #         else:
    #             img_array[0, x, y] = 0
    # img.show()
    prediction = currentmodel.predict(img_array)
    number = np.argmax(prediction)
    print("Number",current,":", number)
    current += 1
