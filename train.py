import tensorflow as tf

import math
import numpy as np

from tensorflow import keras
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.preprocessing import image_dataset_from_directory

from IPython.display import display
import util
import nets
import info

train_ds = image_dataset_from_directory(
    info.data_path,
    batch_size=info.batch_size,
    image_size=(info.crop_size, info.crop_size),
    validation_split=0.2,
    subset="training",
    seed=1337,
    label_mode=None,
)

valid_ds = image_dataset_from_directory(
    info.data_path,
    batch_size=info.batch_size,
    image_size=(info.crop_size, info.crop_size),
    validation_split=0.2,
    subset="validation",
    seed=1337,
    label_mode=None,
)

# Scale from (0, 255) to (0, 1)
train_ds = train_ds.map(util.scaling)
valid_ds = valid_ds.map(util.scaling)

for batch in train_ds.take(1):
    for img in batch:
        display(array_to_img(img))
        

# Use TF Ops to process.
def process_input(input, input_size, upscale_factor):
    input = tf.image.rgb_to_yuv(input)
    last_dimension_axis = len(input.shape) - 1
    y, u, v = tf.split(input, 3, axis=last_dimension_axis)
    return tf.image.resize(y, [input_size, input_size], method="area")


def process_target(input):
    input = tf.image.rgb_to_yuv(input)
    last_dimension_axis = len(input.shape) - 1
    y, u, v = tf.split(input, 3, axis=last_dimension_axis)
    return y

input_size = info.crop_size // info.upscale_factor

train_ds = train_ds.map(
    lambda x: (process_input(x, input_size, info.upscale_factor), process_target(x))
)
train_ds = train_ds.prefetch(buffer_size=32)

valid_ds = valid_ds.map(
    lambda x: (process_input(x, input_size, info.upscale_factor), process_target(x))
)
valid_ds = valid_ds.prefetch(buffer_size=32)

for batch in train_ds.take(1):
    for img in batch[0]:
        display(array_to_img(img))
    for img in batch[1]:
        display(array_to_img(img))
        
"""
The ESPCNCallback object will compute and display the PSNR metric. 
This is the main metric we use to evaluate super-resolution performance.
"""
class ESPCNCallback(keras.callbacks.Callback):
    def __init__(self, filepath=""):
        super(ESPCNCallback, self).__init__()
        #self.test_img = util.get_lowres_image(load_img(test_img_paths[0]), info.upscale_factor)
        #self.filepath = filepath

    # Store PSNR value in each epoch.
    def on_epoch_begin(self, epoch, logs=None):
        self.psnr = []

    def on_epoch_end(self, epoch, logs=None):
        print("Mean PSNR for epoch: %.2f" % (np.mean(self.psnr)))
        #if epoch % 20 == 0:
        #    prediction = util.upscale_image(self.model, self.test_img)
        #    util.plot_results(prediction, self.filepath + "/epoch-" + str(epoch), "prediction")

    def on_test_batch_end(self, batch, logs=None):
        self.psnr.append(10 * math.log10(1 / logs["loss"]))
        
early_stopping_callback = keras.callbacks.EarlyStopping(monitor="loss", patience=10)

model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=info.checkpoint_filepath,
    save_weights_only=True,
    monitor="loss",
    mode="min",
    save_best_only=True,
)

model = nets.get_model(upscale_factor=info.upscale_factor, channels=1)
model.summary()

#callbacks = [ESPCNCallback(info.image_filepath), early_stopping_callback, model_checkpoint_callback]
callbacks = [ESPCNCallback(), early_stopping_callback, model_checkpoint_callback]

model.fit(train_ds, epochs=info.epochs, callbacks=callbacks, validation_data=valid_ds, verbose=2)
