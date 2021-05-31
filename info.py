import os

crop_size = 300
upscale_factor = 3
batch_size = 8
epochs = 100
lr = 0.001

# Download dataset from "http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz"
# Untar the downloaded file
data_path = "BSR/BSDS500/data"  # This has a sub-folder("images"), the sub-folder has 3 sub-folders("train", "val", and "test")
test_path = os.path.join(data_path, "images/test")
file_ext = ".jpg"

checkpoint_filepath = "mmmm/checkpoint"
output_filepath = "real_train/output"

if not os.path.exists(output_filepath):
    os.makedirs(output_filepath)