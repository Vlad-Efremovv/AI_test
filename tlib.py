import matplotlib.pyplot as plt
# matplotlib.use('TkAgg')
import numpy as np
import PIL
import tensorflow as tf
import os

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib 

gpus = tf.config.list_physical_devices('GPU')
if not gpus:
    print("GPU не найден")
else:
    print(f"Найдено {len(gpus)} GPU: {gpus}")

# Проверка доступности GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Ограничение на использование памяти GPU (по желанию)
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Используемая видеокарта: {gpus}")
    except RuntimeError as e:
        print(e)
else:
    print("GPU не найден, будет использоваться CPU.")

dataset_dir = pathlib.Path("C:/Down/flower_photos/flower_photos")

image_count = len(list(dataset_dir.glob("*/*.jpg")))
print(f"Всего изображений:{image_count}")

#количестко датасетов за раз
batch_size = 64

img_width = 180
img_height = 180

train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

# Получение имен классов
class_names = train_ds.class_names
print(f"Class names: {class_names}")

# Кэширование
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# create model
num_classes = len(class_names)
model = Sequential([
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    
    # Аугментация
    layers.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.2),

    # Дальше везде одинаково
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),

    # Регуляризация
    layers.Dropout(0.2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

# compile the model
model.compile(
	optimizer='adam',
	loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	metrics=['accuracy'])

# print model summary
model.summary()

# train the model
epochs = 10 # количество эпох тренировки
history = model.fit(
	train_ds,
	validation_data=val_ds,
	epochs=epochs)

# visualize training and validation results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Путь для сохранения весов
file_path = r"C:\Users\vlade\Documents\my_model.weights.h5"  # Используйте r'' для "сырого" строкового формата, или замените \ на \\

# Сохраняем веса модели
model.save_weights(file_path)

# Получаем полный путь к файлу
full_path = os.path.abspath(file_path)
print('Model saved at:', full_path)
