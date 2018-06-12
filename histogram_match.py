import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize, imsave
import sys
import matplotlib.pyplot as plt


def remap_histogram(image, style):
    return tf.stack([_remap_histogram_level(image[:,:,i], style[:,:,i]) for i in range(3)], axis=-1)

def get_hist_from_img(img):
    img_linearized = tf.reshape(img, (-1,))
    bincounts = tf.bincount(img_linearized, minlength=256)
    bincounts = bincounts / img_linearized.get_shape()
    return tf.cumsum(bincounts)

def cond(i,j,a,b,x):
    return tf.less(i,256)

def body(i,j,a,b,x):
    def false_fn(i,j,x):
        x = tf.concat([x, tf.reshape(j, (1,))], 0)
        i = tf.add(i,1)
        return [i,j,x]
    def true_fn(i,j,x):
        j = tf.add(j, 1)
        return [i,j,x]
    
    i,j,x = tf.cond(tf.logical_and(tf.greater(a[i],b[j]), tf.less(j,255)), lambda: true_fn(i,j,x), lambda: false_fn(i,j,x))
    return i,j,a,b,x

def get_map_fn(mapping):
    def map_fn(elem):
        return tf.gather(mapping, elem)
    return map_fn

def _remap_histogram_level(image, style):
    image = tf.cast(image, tf.int32)
    style = tf.cast(style, tf.int32)

    image_histogram = get_hist_from_img(image)
    style_histogram = get_hist_from_img(style)


    i = tf.constant(0)
    j = tf.constant(0)
    x = tf.constant([], dtype=tf.int32)

    i,j,_,_,hist_mapping = tf.while_loop(
            cond=cond, 
            body=body, 
            loop_vars=[i,j,image_histogram,style_histogram,x], 
            shape_invariants=[i.get_shape(), j.get_shape(), image_histogram.get_shape(), style_histogram.get_shape(), tf.TensorShape([None,])]
            )

    remapped_img = tf.map_fn(get_map_fn(hist_mapping), tf.reshape(image, (-1,)))

    return tf.reshape(remapped_img, image.get_shape())


def _load_image(filename, size=None):
    img = imread(filename, mode='RGB')
    h, w, _ = img.shape
    if size is not None:
        img = imresize(img, size)
    return img

def main():
    if len(sys.argv) < 3:
        print("Usage: ", sys.argv[0], "<input image> <style image> [output name]")
        sys.exit(0)

    path1 = sys.argv[1]
    path2 = sys.argv[2]

    outname = 'out.png' if len(sys.argv) <= 3 else sys.argv[3]

    im1 = _load_image(path1)
    im2 = _load_image(path2)

    im1_tf = tf.constant(im1, dtype=tf.int32)
    im2_tf = tf.constant(im2, dtype=tf.int32)

   

    im3_tf = remap_histogram(im1_tf, im2_tf)

    with tf.Session() as sess:
        im3 = sess.run(im3_tf)
    
    im3 = im3.astype(np.uint8)
    plt.imsave(outname, im3, format=outname[:-3])

if __name__ == '__main__':
    main()