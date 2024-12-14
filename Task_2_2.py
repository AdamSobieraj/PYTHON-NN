import pandas as pd
from keras import Model, Sequential
from keras.regularizers import l2

from keras.applications import VGG16
from keras.layers import Dense, Dropout
from keras.src.callbacks import EarlyStopping
from keras.src.layers import GlobalAveragePooling2D, Conv2D, MaxPooling2D, Flatten
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.optimizers import RMSprop

from neural_net_2 import train_data_dir, img_height, img_width, batch_size, validation_data_dir, train_size, valid_size, \
     train_generator_augmentation, steps_per_epoch, epochs, validation_steps, es

# ... (previous imports remain the same)

def create_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

    # Freeze all layers except the last block5_conv1
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=x)

    return model


model = create_model()
model.compile(optimizer=RMSprop(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    validation_split=0.2)

# Split data into training and validation sets
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='training')

validation_generator = train_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation')

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

history = model.fit(train_generator,
                    steps_per_epoch=train_size // batch_size,
                    epochs=100,
                    validation_data=validation_generator,
                    validation_steps=valid_size // batch_size,
                    callbacks=[early_stopping])

# Save the best model
model.save('best_model.h5')
models = []

# Model 5: Enhanced model with more convolutional layers and L2 regularization
model_5 = Sequential()
model_5.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model_5.add(Conv2D(64, (3, 3), activation='relu'))
model_5.add(MaxPooling2D((2, 2)))
model_5.add(Conv2D(128, (3, 3), activation='relu'))
model_5.add(Conv2D(256, (3, 3), activation='relu'))
model_5.add(MaxPooling2D((2, 2)))
model_5.add(Flatten())
model_5.add(Dense(512, activation='relu', kernel_regularizer=l2(0.01)))
model_5.add(Dense(256, activation='relu', kernel_regularizer=l2(0.01)))
model_5.add(Dense(1, activation='sigmoid'))
model_5.compile(optimizer=RMSprop(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
model_5.summary()
models.append("model_5")

# Model 6: Enhanced model with dropout regularization
model_6 = Sequential()
model_6.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model_6.add(Conv2D(64, (3, 3), activation='relu'))
model_6.add(MaxPooling2D((2, 2)))
model_6.add(Conv2D(128, (3, 3), activation='relu'))
model_6.add(Conv2D(256, (3, 3), activation='relu'))
model_6.add(MaxPooling2D((2, 2)))
model_6.add(Flatten())
model_6.add(Dense(512, activation='relu'))
model_6.add(Dropout(0.5))
model_6.add(Dense(256, activation='relu'))
model_6.add(Dropout(0.5))
model_6.add(Dense(1, activation='sigmoid'))
model_6.compile(optimizer=RMSprop(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
model_6.summary()
models.append("model_6")

# Train models 5 and 6
history_model_5 = model_5.fit_generator(train_generator_augmentation,
                                        steps_per_epoch=steps_per_epoch,
                                        epochs=epochs,
                                        validation_data=validation_generator,
                                        validation_steps=validation_steps,
                                        callbacks=[es])

history_model_5_df = pd.DataFrame(history_model_5.history)
history_model_5_csv_file = 'history/history_model_5.csv'
with open(history_model_5_csv_file, mode='w') as f:
    history_model_5_df.to_csv(f)

history_model_6 = model_6.fit_generator(train_generator_augmentation,
                                        steps_per_epoch=steps_per_epoch,
                                        epochs=epochs,
                                        validation_data=validation_generator,
                                        validation_steps=validation_steps,
                                        callbacks=[es])

history_model_6_df = pd.DataFrame(history_model_6.history)
history_model_6_csv_file = 'history/history_model_6.csv'
with open(history_model_6_csv_file, mode='w') as f:
    history_model_6_df.to_csv(f)