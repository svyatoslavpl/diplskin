import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import concatenate, Activation, Dropout, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from utils.data_processing import create_train_test_split, prepare_data
from models.soft_attention import SoftAttention


def train_model():
    train_dir = 'data/HAM10000/train_dir'
    test_dir = 'data/HAM10000/test_dir'

    data_pd = pd.read_csv('data/HAM10000_metadata.csv')

    train_df, test_df = create_train_test_split(data_pd, test_df, train_dir)

    train_list = list(train_df['image_id'])
    test_list = list(test_df['image_id'])
    prepare_data(train_list, test_list, data_pd, targetnames, train_dir, test_dir)

    batch_size = 16
    datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.inception_resnet_v2.preprocess_input)
    image_size = 224

    print("\nTrain Batches:")
    train_batches = datagen.flow_from_directory(directory=train_dir,
                                                target_size=(image_size, image_size),
                                                batch_size=batch_size,
                                                shuffle=True)

    print("\nTest Batches:")
    test_batches = datagen.flow_from_directory(test_dir,
                                               target_size=(image_size, image_size),
                                               batch_size=batch_size,
                                               shuffle=False)

    resnet = ResNet50(include_top=True, weights="imagenet", input_tensor=None, input_shape=None, pooling=None,
                      classes=1000)
    conv = resnet.layers[-3].output
    attention_layer, map2 = SoftAttention(aggregate=True, m=16, concat_with_x=False,
                                          ch=int(conv.shape[-1]), name='soft_attention')(conv)
    attention_layer = (MaxPooling2D(pool_size=(2, 2), padding="same")(attention_layer))
    conv = (MaxPooling2D(pool_size=(2, 2), padding="same")(conv))
    conv = concatenate([conv, attention_layer])
    conv = Activation('relu')(conv)
    conv = Dropout(0.5)(conv)
    output = GlobalAveragePooling2D()(conv)
    output = Dense(7, activation='softmax')(output)
    model = Model(inputs=resnet.input, outputs=output)
    model.summary()

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    checkpoint = ModelCheckpoint("models/ResNet50+SA.hdf5", monitor='val_accuracy', verbose=1, save_best_only=True,
                                 mode='max')
    early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    callbacks_list = [checkpoint, early_stop]

    model.fit(train_batches, steps_per_epoch=len(train_df) // batch_size, epochs=10, verbose=1,
              callbacks=callbacks_list, validation_data=test_batches, validation_steps=len(test_df) // batch_size)

    model.save("models/ResNet50+SA.h5")
