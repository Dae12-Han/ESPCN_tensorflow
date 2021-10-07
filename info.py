import os

crop_size = 255
upscale_factor = 3
batch_size = 8
epochs = 50
lr = 0.001

# Download dataset from "http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz"
# Untar the downloaded file
data_path = "D:/dataset/DIV2K"  # This has a sub-folder("images"), the sub-folder has 3 sub-folders("train", "val", and "test")
#test_path = os.path.join(data_path, "test")
#test_path = "D:/dataset/valid/test_original"
test_path = "D:/dataset/CG_test"
file_ext = (".jpg",".png")

checkpoint_filepath = "DIV2K/checkpoint"
output_filepath = "DIV2K/CG_test/output"

if not os.path.exists(output_filepath):
    os.makedirs(output_filepath)