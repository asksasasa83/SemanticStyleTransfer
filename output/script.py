from PIL import Image

import os
import glob

WIDTH = 500


OUTDIR = os.path.abspath('./ALL')
os.makedirs(OUTDIR, exist_ok=True)

style = ['conv1_1,conv2_1,conv3_1,conv4_1,conv5_1', 'conv1_2,conv2_2,conv3_4,conv4_4,conv5_4', 'conv1_2,conv2_2,conv3_2,conv4_2,conv5_2', 'conv3_1,conv4_1,conv5_1', 'conv3_1,conv4_1,conv5_1']
content = ['conv3_2', 'conv4_2', 'conv5_2']
tv = '0.00001'
style_weights=['5','100', '500']


for i,cont in enumerate(content):
    for j,sty in enumerate(style):
        for style_weight in style_weights:
            for algo in ['Adam', 'L-BFGS']:
                for init in ['image', 'random']:
                    path = os.path.join(cont, sty, style_weight, algo, init, tv, 'tubingen_starry-night.jpg')

                    im = Image.open(path)
                    im = im.resize((WIDTH, int(WIDTH / im.size[0] * im.size[1])))

                    saved_path = os.path.join(OUTDIR, '_'.join(map(str, [i, j, style_weight, algo, init]))) + '.jpg'
                    im.save(saved_path)
