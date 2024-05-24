import os
import yaml


base_dir = os.path.abspath(os.path.join(__file__, "..", "..", ".."))
original_path = "dataset/leftImg8bit_trainvaltest/leftImg8bit/train"
output_path = "output"

left_image_path = os.path.join(base_dir, original_path)
size = 512

# open yaml
def read_yaml():
    try:
        with open("clean_data.yaml") as stream:
            return yaml.safe_load(stream)
    except FileNotFoundError:
        write_yaml({})
        return read_yaml()

data = read_yaml()

output = {}
for city, images in data.items():
    for image, t in images.items():
        if t not in output:
            output[t] = 1
        else:
            output[t] += 1

print(output)