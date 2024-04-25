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
                    continue
                c_i = v2.Resize(size=size)(read_image(l_image_path, mode=ImageReadMode.RGB))
                
                def output_image(t):
                    return v2.Resize(size=size)(read_image(os.path.join(base_dir, output_path, t, city, clean_name), mode=ImageReadMode.RGB))
                # comb
                comb = output_image("combined")
                hor = output_image("horisontal")
                ver = output_image("vertical")
                comb_50 = output_image("combined_50")
                hor_50 = output_image("horisontal_50")
                ver_50 = output_image("vertical_50")

                fix, axs = pyplot.subplots(ncols=7, squeeze=False)
                for i, img in enumerate([c_i, comb, hor, ver, comb_50, hor_50, ver_50]):
                    img = img.detach()
                    img = torchvision.transforms.functional.to_pil_image(img)
                    axs[0, i].imshow(numpy.asarray(img))
                    axs[0, i].title.set_text(str(i))

                    axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
                fix.set_figwidth(50)
                fix.set_figheight(10)

                

                
                if city not in data:
                    data[city] = {}

                pyplot.show(block=False)
                # pyplot.figure()
                i = input("1,2,3,4,5,6,0")
                pyplot.close()
                
                if i.strip() == "exit":
                    import sys
                    sys.exit(-1)
                elif int(i.strip()) == 0:
                    data[city][clean_name] = "failure"
                elif int(i.strip()) == 1:
                    data[city][clean_name] = "combined"
                elif int(i.strip()) == 2:
                    data[city][clean_name] = "horisontal"
                elif int(i.strip()) == 3:
                    data[city][clean_name] = "vertical"
                elif int(i.strip()) == 4:
                    data[city][clean_name] = "combined_50"
                elif int(i.strip()) == 5:
                    data[city][clean_name] = "horisontal_50"
                elif int(i.strip()) == 6:
                    data[city][clean_name] = "vertical_50"

                write_yaml(data)
# wait input

# Write yaml

# next

