import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config
import sys


def main(origin_dir):
    image_names = [files for root, dirs, files in os.walk(origin_dir)][0]
    print('find %s files in %s' % (len(image_names), origin_dir))

    tflib.init_tf()
    _G, D, _Gs = pickle.load(open(config.Model, "rb"))

    for index, image_name in enumerate(image_names):
        image_path = os.path.join(origin_dir, image_name)
        img = np.asarray(PIL.Image.open(image_path))
        img = img.reshape(1, 3, 512, 512)
        score = D.run(img, None)
        os.rename(image_path, os.path.join(origin_dir, '%s_%s.png' % (score[0][0], index)))
        print(image_name, score[0][0])

    print('Done!')


if __name__ == "__main__":
    main(sys.argv[1])
