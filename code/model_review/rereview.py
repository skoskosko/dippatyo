import os
import matplotlib
# matplotlib.use('TkAgg')
from matplotlib import pyplot
from torchvision.transforms import v2
from torchvision.io import read_image
from torchvision.utils import make_grid
from torchvision.io import ImageReadMode
import torchvision.transforms.functional
import numpy
import yaml
import io



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

def write_yaml(data):
    with io.open('clean_data.yaml', 'w', encoding='utf8') as outfile:
        yaml.dump(data, outfile, default_flow_style=False, allow_unicode=True)


data = read_yaml()
for city in os.listdir(left_image_path):
    city_dir = os.path.join(left_image_path, city)
    if os.path.isdir(city_dir):
        for l_image in os.listdir(city_dir):
            if l_image.endswith(".png"):
                l_image_path = os.path.join(city_dir, l_image)
                clean_name = l_image.replace("_leftImg8bit", "")
                if city in data and clean_name in data[city]:
                    if "version" in data[city][clean_name] and data[city][clean_name]["version"] == "v2":
                        continue
                
                c_i = v2.Resize(size=size)(read_image(l_image_path, mode=ImageReadMode.RGB))
                
                def output_image(t):
                    try:
                        return v2.Resize(size=size)(read_image(os.path.join(base_dir, output_path, t, city, clean_name), mode=ImageReadMode.RGB))
                    except:
                        return v2.Resize(size=size)(read_image("/home/esko/Documents/Dippatyo/output/placeholder.png", mode=ImageReadMode.RGB))
                # comb

                old = output_image(data[city][clean_name])
                focal = output_image("focal")

                fix, axs = pyplot.subplots(ncols=3, squeeze=False)
                for i, img in enumerate([c_i, old, focal]):
                    img = img.detach()
                    img = torchvision.transforms.functional.to_pil_image(img)
                    axs[0, i].imshow(numpy.asarray(img))
                    axs[0, i].title.set_text(str(i+1))

                    axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
                fix.set_figwidth(50)
                fix.set_figheight(10)

                if city not in data:
                    data[city] = {}

                pyplot.show(block=False)
                # pyplot.figure()
                i = input("1,2,3")
                pyplot.close()
                
                if i.strip() == "exit":
                    import sys
                    sys.exit(-1)
                elif int(i.strip()) == 1:
                    data[city][clean_name] = {"version": "v2"}
                    data[city][clean_name]["name"] = "failure"
                elif int(i.strip()) == 2:
                    name = data[city][clean_name]
                    data[city][clean_name] = {"version": "v2"}
                    data[city][clean_name]["name"] = name
                elif int(i.strip()) == 3:
                    data[city][clean_name] = {"version": "v2"}
                    data[city][clean_name]["name"] = "focal"
                else:
                    raise Exception("NOT DETECTED!!!")

                write_yaml(data)
