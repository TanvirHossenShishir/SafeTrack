import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Conv2D, BatchNormalization, MaxPooling2D,Dropout, Flatten
from sklearn.model_selection import train_test_split

Data = "/content/data/train/"
Classes = ["close eyes", "open eyes"]

X = []
y = []
data_limit = 24000
img_size = 64

def create_training_data():
    count = 0
    category_count = [0, 0]
    for category in Classes:
        path = os.path.join(Data, category)
        class_num = Classes.index(category)
        for img in os.listdir(path):
            try:
                if count >= data_limit:
                    return

                if category_count[class_num] >= data_limit // 2:
                    continue

                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (img_size, img_size))
                new_array = np.expand_dims(new_array, axis=-1)

                X.append(new_array)
                y.append(class_num)

                count += 1
                category_count[class_num] += 1
            except Exception as e:
                print(e)

create_training_data()

X = np.array(X)
X = X / 255.0
X = (np.array(X) - np.min(X)) / (np.max(X) - np.min(X))
Y = np.array(y)

model = tf.keras.models.Sequential([
      Input(shape=(64, 64, 1)),

      Conv2D(filters = 32, kernel_size = 5, strides = 1, activation = 'relu'),
      Conv2D(filters = 32, kernel_size = 5, strides = 1, activation = 'relu', use_bias=False),
      BatchNormalization(),
      MaxPooling2D(strides = 2),
      Dropout(0.3),

      Conv2D(filters = 64, kernel_size = 3, strides = 1, activation = 'relu'),
      Conv2D(filters = 64, kernel_size = 3, strides = 1, activation = 'relu', use_bias=False),
      BatchNormalization(),
      MaxPooling2D(strides = 2),
      Dropout(0.3),

      Flatten(),
      Dense(units  = 256, activation = 'relu', use_bias=False),
      BatchNormalization(),

      Dense(units = 128, use_bias=False, activation = 'relu'),

      Dense(units = 84, use_bias=False, activation = 'relu'),
      BatchNormalization(),
      Dropout(0.3),

      Dense(units = 1, activation = 'sigmoid')
  ])

model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='model.h5',
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    verbose =1)
model.fit(x_train, y_train, validation_split=0.2, epochs=30, batch_size=32, callbacks=callback)

model.evaluate(x_test, y_test)