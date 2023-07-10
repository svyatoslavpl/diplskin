import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Activation, Dropout, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

def train_model_without_SA():
    train_dir = 'data/HAM10000/train_dir'
    test_dir = 'data/HAM10000/test_dir'

    # Остальной код остается без изменений

    resnet = ResNet50(include_top=True, weights="imagenet", input_tensor=None, input_shape=None, pooling=None,
                      classes=1000)
    conv = resnet.layers[-3].output
    conv = Activation('relu')(conv)
    conv = Dropout(0.5)(conv)
    output = GlobalAveragePooling2D()(conv)
    output = Dense(7, activation='softmax')(output)
    model = Model(inputs=resnet.input, outputs=output)
    model.summary()

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # Остальной код остается без изменений

    model.fit(train_batches, steps_per_epoch=len(train_df) // batch_size, epochs=10, verbose=1,
              callbacks=callbacks_list, validation_data=test_batches, validation_steps=len(test_df) // batch_size)

    model.save("models/ResNet50_without_SA.h5")
