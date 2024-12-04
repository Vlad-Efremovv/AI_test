import tensorflow as tf

gpus = tf.config.list_physical_devices()
if gpus:
    print("Num GPUs Available: ", len(gpus))
    print("Available GPUs: ", gpus[0].name)
else:
    print("GPU не найден")
