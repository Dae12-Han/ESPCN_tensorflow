import os
import sys
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

import info
import nets
import util

model = nets.get_model(upscale_factor=info.upscale_factor, channels=1)
model.summary()

# The model weights (that are considered the best) are loaded into the model.
model.load_weights(info.checkpoint_filepath)

test_img_paths = sorted(
    [
        os.path.join(info.test_path, fname)
        for fname in os.listdir(info.test_path)
        if fname.endswith(info.file_ext)
    ]
)

# Run model prediction and plot the results
total_bicubic_psnr = 0.0
total_test_psnr = 0.0

def get_fname(file_path):
    return file_path.split("\\")[-1]

sub_test_img_paths = test_img_paths[50:60]
for index, test_img_path in enumerate(sub_test_img_paths):
    img = load_img(test_img_path)
    lowres_input = util.get_lowres_image(img, info.upscale_factor)
    w = lowres_input.size[0] * info.upscale_factor
    h = lowres_input.size[1] * info.upscale_factor
    highres_img = img.resize((w, h))
    prediction = util.upscale_image(model, lowres_input) # superresolution
    lowres_img = lowres_input.resize((w, h))    # bicubic interpolation
    lowres_img_arr = img_to_array(lowres_img)
    highres_img_arr = img_to_array(highres_img)
    predict_img_arr = img_to_array(prediction)
    bicubic_psnr = tf.image.psnr(lowres_img_arr, highres_img_arr, max_val=255)
    test_psnr = tf.image.psnr(predict_img_arr, highres_img_arr, max_val=255)

    total_bicubic_psnr += bicubic_psnr
    total_test_psnr += test_psnr

    sys.stdout=open("CG_test.txt",'a')
    print("%s: PSNR (BICUBIC = %.4f, ESPCN = %.4f)" % (get_fname(test_img_path), bicubic_psnr, test_psnr))
    sys.stdout.close()

    util.plot_results(lowres_img, info.output_filepath + "/" + get_fname(test_img_path).rstrip(".jpg"), "BC", False)  # bicubic result
    util.plot_results(highres_img, info.output_filepath + "/" + get_fname(test_img_path).rstrip(".jpg"), "HR", False)
    util.plot_results(prediction, info.output_filepath + "/" + get_fname(test_img_path).rstrip(".jpg"), "SR", False)  # SR result

sys.stdout=open("CG_test.txt",'a')
print("Mean: PSNR (BICUBIC = %.4f, ESPCN = %.4f)" % (total_bicubic_psnr / len(sub_test_img_paths), total_test_psnr / len(sub_test_img_paths)))
sys.stdout.close()