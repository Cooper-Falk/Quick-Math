import tensorflow as tf
import numpy as np
import cv2
import matplotlib as plt
from PIL import Image
import os

mnist = tf.keras.datasets.mnist
(xtrain, ytrain), (xtest, ytest) = mnist.load_data()
xtrain = tf.keras.utils.normalize(xtrain, axis=1)
xtest = tf.keras.utils.normalize(xtest, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dense(10, activation="softmax"))
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(xtrain, ytrain, epochs=1)
model.save("first.model")
currentmodel = tf.keras.models.load_model("first.model")

##CANT GIVE THE PROPER TYPE OF FILE
image = cv2.imread("number1.png", cv2.IMREAD_GRAYSCALE)
if image.size == 0:
    raise ValueError("Failed to load image file")
# image = cv2.resize(image, (28, 28))
image = cv2.bitwise_not(image)
image = image.astype('float32') / 255.0
image = np.expand_dims(image, axis=-1)
image = np.expand_dims(image, axis=0)
prediction = model.predict(image)
predicted_class = np.argmax(prediction)
print("The predicted digit is:", predicted_class)

# currentnum=1
# # while os.path.isfile(f"numberpictures/number{currentnum}.png"):
# im = cv2.imread("number1.png")
# im = np.invert(np.array([im]))
# # prediction = currentmodel.predict(im)
# # print("This is a: "+np.argmax(prediction))
# # plt.imshow(im, cmap = plt.cm.binary)
# plt.show()
# currentnum+=1


# #I DIDN'T DO IT OPEN CV2 WHATS THE POINT MAN.
# # with Image.open("first.model/numberpictures/number2.png") as im:
#     im = Image.open(f"numerpictures/number{currentnum}.png")
#     width, height = im.size
#     print(width,height)
#     if width > height:
#         diff = width - height
#         width = height
#         (left, upper, right, lower) = ((diff/2), 0, width+(diff/2), height)
#     else:
#         diff = height - width
#         height = width
#         (left, upper, right, lower) = (0, (diff/2),  width, height+(diff/2))
# # im = im.crop(width, height)
#     print((left, upper, right, lower))
#     im = im.crop((left, upper, right, lower))
#     im = im.resize((28, 28),Image.ANTIALIAS)
#     im.show()
#     currentnum+=1
