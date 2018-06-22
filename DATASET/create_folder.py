from PIL import Image

import os
import glob

WIDTH = 200


OUTDIR = os.path.abspath('../OUTDIR')
os.makedirs(OUTDIR, exist_ok=True)

algorithms = ['gatys', 'gatys_histogram', 'dpa1', 'dpa2', 'photo']

for folder in glob.glob('./*/'):
    num = os.path.basename(os.path.dirname(folder))

    cont = Image.open(os.path.join(folder, 'content.jpg'))
    cont = cont.resize((WIDTH, int(WIDTH / cont.size[0] * cont.size[1])))
    
    style = Image.open(os.path.join(folder, 'style.jpg'))
    style = style.resize((WIDTH, int(WIDTH / style.size[0] * style.size[1])))

    for algo in algorithms:
        try:
            im = Image.open(os.path.join(folder, algo, 'output.jpg'))
            im = im.resize((WIDTH, int(WIDTH / im.size[0] * im.size[1])))
            im.save(os.path.join(OUTDIR, algo + '_' + num) + '.jpg')
        except Exception as e:
            print(e)

    
    cont.save(os.path.join(OUTDIR, 'content_' + num) + '.jpg')
    style.save(os.path.join(OUTDIR, 'style_' + num) + '.jpg')