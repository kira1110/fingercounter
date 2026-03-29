import numpy as np
import tensorflow as tf
from tensorflow.keras import layers,models
from sklearn.model_selection import train_test_split

X = np.load("X.npy")
y = np.load("y.npy")

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

model = models.Sequential([
    layers.Conv2D(32,(3,3),activation='relu',input_shape=(64,64,3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128,(3,3),activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128,activation='relu'),
    layers.Dense(6,activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train,y_train,epochs=10,validation_data=(X_test,y_test))

model.save("finger_model.h5")