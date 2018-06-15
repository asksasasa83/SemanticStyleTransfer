from PIL import Image

import os
import glob

for folder in glob.glob('./*/'):
    cont = Image.open(os.path.join(folder, 'content.jpg'))
    style = Image.open(os.path.join(folder, 'style.jpg'))

    style_resize = style.resize(cont.size)
    style_resize.save(os.path.join(folder, 'style.jpg'))

    mask = Image.new('RGB', cont.size, (255, 255, 255))
    mask.save(os.path.join(folder, 'mask.jpg'))
    mask.save(os.path.join(folder, 'mask_dilated.jpg'))