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
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(xtrain, ytrain, epochs=200)
model.save("first.model")
results = model.evaluate(xtest, ytest, batch_size=128)

currentmodel = tf.keras.models.load_model("first.model")
current = 1
while os.path.isfile(f"numberpictures/number{current}.png"):
    img = Image.open(f"numberpictures/number{current}.png").convert("L")
    img = img.resize((28, 28))
    # img.show()
    img_array = np.invert(np.array(img))
    img_array = img_array.reshape(1, 28, 28)
    img_array = img_array / 255.0
    for x in range(28):
        for y in range(28):
            if img_array[0, x, y] > 0:
                img_array[0, x, y] = 1
            else:
                img_array[0, x, y] = 0
    prediction = currentmodel.predict(img_array)
    number = np.argmax(prediction)
    print("This is most likey a", number)
    current += 1

# ott = Image.open("onetwothree.png")
# width,height = ott.size
# print(width,height)

# trigger = True
# newImg = Image.open("onetwothree.png")
# newImg.show()
# while trigger:
#     runner = True
#     start = 0
#     img_array = np.invert(np.array(newImg))
#     img_array = img_array / 255.0
#     for x in range(img_array.shape[1]):
#         for y in range(img_array.shape[0]):
#             if np.any(img_array[y][x] > 0):
#                 if runner:
#                     start = y
#                     runner = False
#     if runner == False:
#         croppedimg = newImg.crop((start-20, 0, start+80, 100))
#         currentImg = croppedimg.save("currentImg.png")
#         currentImg1 = Image.open("currentImg.png")
#         # currentImg1.show()
#         guessNumber()
#     else:
#         trigger = False
#     img_array = img_array[0:start+80, 100:img_array.shape[1]+start+80]
#     newImg = newImg.crop((start+80,0, img_array.shape[1]+start+80, 100))
#     newImg.show()
