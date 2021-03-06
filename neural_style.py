#!/usr/bin/env python
from operator import add
import os

from scipy.misc import imread, imresize, imsave
from tensorflow.contrib.opt import ScipyOptimizerInterface
import h5py
import numpy as np
import tensorflow as tf
import histogram_match


class NeuralStyle:
    VGG19_MEAN_BGR = (103.939, 116.779, 123.68)
    VGG19_NO_FC = (
        ('conv', 'conv1_1', (3, 3, 3, 64)),
        ('conv', 'conv1_2', (3, 3, 64, 64)),
        ('pool', 'pool1',   (1, 2, 2, 1)),
        ('conv', 'conv2_1', (3, 3, 64, 128)),
        ('conv', 'conv2_2', (3, 3, 128, 128)),
        ('pool', 'pool2',   (1, 2, 2, 1)),
        ('conv', 'conv3_1', (3, 3, 128, 256)),
        ('conv', 'conv3_2', (3, 3, 256, 256)),
        ('conv', 'conv3_3', (3, 3, 256, 256)),
        ('conv', 'conv3_4', (3, 3, 256, 256)),
        ('pool', 'pool3',   (1, 2, 2, 1)),
        ('conv', 'conv4_1', (3, 3, 256, 512)),
        ('conv', 'conv4_2', (3, 3, 512, 512)),
        ('conv', 'conv4_3', (3, 3, 512, 512)),
        ('conv', 'conv4_4', (3, 3, 512, 512)),
        ('pool', 'pool4',   (1, 2, 2, 1)),
        ('conv', 'conv5_1', (3, 3, 512, 512)),
        ('conv', 'conv5_2', (3, 3, 512, 512)),
        ('conv', 'conv5_3', (3, 3, 512, 512)),
        ('conv', 'conv5_4', (3, 3, 512, 512)),
        ('pool', 'pool5',   (1, 2, 2, 1))
    )


    def __init__(self, *,
            content_image='content.jpg',
            content_layers='conv4_2',
            content_weight=5e0,
            image_size=500,
            init: 'random|image' = 'random',
            num_iterations=500,
            output_image='output.jpg',
            print_iter=50,
            save_iter=100,
            style_image='style.jpg',
            style_layers='conv1_1,conv2_1,conv3_1,conv4_1,conv5_1',
            style_weight=1e2,
            tv_weight=1e-3,
            optimizer='Adam',
            hist_weight=1e-3):
        self._content_image = content_image
        self._content_layers = content_layers.split(',')
        self._content_weight = content_weight
        self._init = init
        self._num_iterations = num_iterations
        self._image_size = image_size
        self._output_image = output_image
        self._print_iter = print_iter
        self._save_iter = save_iter
        self._style_image = style_image
        self._style_layers = style_layers.split(',')
        self._style_weight = style_weight
        self._tv_weight = tv_weight
        self._nodes = {}
        self._step_counter = 0
        self._optimizer = optimizer
        self._hist_weight = hist_weight


    def run(self):
        content = self._load_image(self._content_image)
        h,w = content.shape[1], content.shape[2]
        style = self._load_image(self._style_image, size=(h,w))

        print("Content shape: ", content.shape)
        print("Style shape: ", style.shape)

        image = tf.Variable(style,
            dtype=tf.float32, validate_shape=False, name='image')
        self._output_shape = content.shape
        self._build_vgg19(image)
        self._add_gramians()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            # Calculate loss function
            sess.run(tf.global_variables_initializer())
            with tf.name_scope('losses'):
                style_losses = self._setup_style_losses(sess, image, style)
                content_losses = self._setup_content_losses(sess, image, content)
                
                losses = content_losses + style_losses
                
                if self._hist_weight > 0:
                    with tf.name_scope('histogram'), tf.device('/cpu:0'):
                        hist_loss = self._setup_histogram_loss(image, style, sess)
                    losses += hist_loss

                image.set_shape(content.shape) # tv loss expects explicit shape
                if self._tv_weight:
                    tv_loss = tf.image.total_variation(image[0])
                    tv_loss_weighted = tf.multiply(tv_loss, self._tv_weight,
                        name='tv_loss')
                    losses += tv_loss_weighted
                loss = tf.foldl(add, losses, name='loss')

            # Set optimizator
            if self._optimizer == 'Adam':
                opt = tf.train.AdamOptimizer(10).minimize(loss)
            
                sess.run(tf.global_variables_initializer())
                self._set_initial_image(sess, image, content)

                self._step_callback(sess.run(image))

                for it in range(self._num_iterations):
                    _,ll, out = sess.run([opt, loss, image])
                    self._step_callback(out)
                    print("Iteration: {:3d}\tLoss = {:.6f}".format(it, ll))
            
            elif self._optimizer == 'L-BFGS':
                sess.run(tf.global_variables_initializer())
                self._set_initial_image(sess, image, content)
                self._step_callback(sess.run(image))

                opt = ScipyOptimizerInterface(loss, options={
                        'maxiter': self._num_iterations,
                        'disp': self._print_iter}, method='L-BFGS-B')
                opt.minimize(sess, step_callback=self._step_callback)
            else:
                raise ValueError("Unknown optimization method")

            self._save_image(self._output_image, sess.run(image))

    def _set_initial_image(self, sess, image, content):
        if self._init == 'image':
            sess.run(tf.assign(image, content, validate_shape=False))
        elif self._init == 'random':
            noise = tf.random_normal(content.shape, stddev=0.001)
            sess.run(tf.assign(image, noise, validate_shape=False))
        else:
            raise ValueError("Unknown init method: " + self._init)

    def _build_vgg19(self, prev):
        weights = KerasVGG19Weights('model/vgg19_weights.h5')
        vgg_trunc = NeuralStyle.VGG19_NO_FC[0:self._vgg_last_useful_layer()+1]
        with tf.name_scope('vgg19_truncated'):
            for layer_type, name, shape in vgg_trunc:
                if layer_type == 'conv':
                    prev = self._conv(prev, name, shape, weights)
                elif layer_type == 'pool':
                    prev = self._pool(prev, name, shape)
                else:
                    raise ValueError('Unknown layer: %s' % layer_type)


    def _add_gramians(self):
        with tf.name_scope('gramians'):
            for style_layer_name in self._style_layers:
                name = 'gramian_' + style_layer_name
                node = self._nodes[style_layer_name]
                with tf.name_scope('gramian'):
                    shape = tf.shape(node) # NxHxWxC
                    h,w = shape[1]*shape[2], shape[3]
                    flat = tf.reshape(node, tf.stack([h,w]))
                    gramian = tf.matmul(flat, flat, transpose_a=True)
                    norm_gramian = gramian / tf.cast(h*w, dtype=tf.float32)
                self._nodes[name] = norm_gramian


    def _vgg_last_useful_layer(self):
        useful_layers = self._style_layers + self._content_layers
        vgg_layers = [name for _, name, _ in NeuralStyle.VGG19_NO_FC]
        return max(vgg_layers.index(layer) for layer in useful_layers)

    def _setup_histogram_loss(self, image, style, sess):
        image_clipped = tf.clip_by_value(tf.cast(image[0] + NeuralStyle.VGG19_MEAN_BGR, tf.int32), 0, 255)
        style_clipped = tf.clip_by_value(tf.cast(style[0] + NeuralStyle.VGG19_MEAN_BGR, tf.int32), 0, 255)

        matched = histogram_match.remap_histogram(image_clipped, style_clipped)
        matched = tf.cast(matched, tf.float32)
        loss = tf.reduce_mean(tf.squared_difference(matched, image))
        loss = tf.multiply(loss, self._hist_weight)
        return [loss]

    def _setup_style_losses(self, sess, image, image_data):
        losses = []
        with tf.name_scope('style_losses'):
            gramians = [self._nodes['gramian_'+l] for l in self._style_layers]
            activations = sess.run(gramians, {image: image_data})
            losses = []
            for i, l in enumerate(self._style_layers):
                prediction = gramians[i]
                target = tf.constant(activations[i], name='const_gramian_'+l)
                loss = tf.reduce_mean(tf.squared_difference(prediction, target))
                loss = tf.multiply(loss, self._style_weight)
                losses.append(loss)
        return losses


    def _setup_content_losses(self, sess, image, image_data):
        losses = []
        with tf.name_scope('content_losses'):
            layers = [self._nodes[l] for l in self._content_layers]
            activations = sess.run(layers, {image: image_data})
            for i, l in enumerate(self._content_layers):
                prediction = self._nodes[l]
                target = tf.constant(activations[i], name='const_'+l)
                loss = tf.reduce_mean(tf.squared_difference(prediction, target))
                loss = tf.multiply(loss, self._content_weight)
                losses.append(loss)
        return losses


    def _conv(self, prev, name, shape, weights):
        with tf.name_scope(name):
            W = tf.Variable(weights[name+'_W'], trainable=False, name='weights')
            b = tf.Variable(weights[name+'_b'], trainable=False, name='biases')
            conv2d = tf.nn.conv2d(prev, W, (1,1,1,1), padding='SAME')
            output = tf.nn.relu(tf.nn.bias_add(conv2d, b), name=name)
        self._nodes[name] = output
        return output


    def _pool(self, prev, name, shape):
        with tf.name_scope(name):
            output = tf.nn.max_pool(prev,
                ksize=shape, strides=shape, padding='SAME', name=name)
        self._nodes[name] = output
        return output


    def _step_callback(self, variables_vector): 
        if (self._step_counter % self._save_iter == 0):
            root, ext = os.path.splitext(self._output_image)
            filename = '%s_%04d%s' % (root, self._step_counter, ext)
            image = np.copy(variables_vector).reshape(self._output_shape)
            self._save_image(filename, image)
        
        self._step_counter += 1


    def _load_image(self, filename, size=None):
        img = imread(filename, mode='RGB')
        (h, w, _), m = img.shape, self._image_size
        if size is not None:
            img = imresize(img, size)
        elif max(img.shape) > m:
            h, w = (m, round(m*w/h)) if h > w else (round(m*h/w), m)
            img = imresize(img, (h, w))
        img = img.astype(np.float32)
        img = np.flip(img, axis=2) # RGB --> BGR
        img -= NeuralStyle.VGG19_MEAN_BGR
        return img[np.newaxis,:]  # conv layers expect 4D-tensors


    def _save_image(self, filename, img):
        img = img[0,:]
        img += NeuralStyle.VGG19_MEAN_BGR
        img = np.flip(img, axis=2)
        img = np.clip(img, 0, 255)
        imsave(filename, img.astype(np.uint8))


class KerasVGG19Weights:


    def __init__(self, filename):
        filename = os.path.realpath(
            os.path.join(os.getcwd(),
            os.path.dirname(__file__),
            filename))
        self._f = h5py.File(filename, 'r')


    def __getitem__(self, name):
        block, name = self._map_name(name)
        return np.array(self._f[block][name])


    def _map_name(self, name):
        "conv1_1_W --> ('block1_conv1', 'block1_conv1_W1:0')"
        block, idx, _type = name.split('_')
        block_idx = block[4:]
        return (
            'block%s_conv%s' % (block_idx, idx),
            'block%s_conv%s_%s_1:0' % (block_idx, idx, _type))


if __name__ == '__main__':
    import inspect
    import sys

    sig = inspect.signature(NeuralStyle.__init__)
    params = [(name, param.default, param.annotation)
                for name, param in sig.parameters.items()
                if param.kind == inspect.Parameter.KEYWORD_ONLY]

    flags = {
        bool: tf.flags.DEFINE_boolean,
        float: tf.flags.DEFINE_float,
        int: tf.flags.DEFINE_integer,
        str: tf.flags.DEFINE_string
    }
    for name, default, annotation in params:
        docstring = '[%(default)s]'
        if annotation != inspect.Parameter.empty:
            docstring = annotation + ' ' + docstring
        flags[type(default)](name, default, docstring)

    def launcher(args):
        if len(args) > 1:
            print("Unknown arguments provided:", " ".join(args[1:]))
            print("Launch with -h flag for help")
            exit(1)
        values = {name: getattr(tf.flags.FLAGS, name) for name, _, _ in params}
        NeuralStyle(**values).run()

    tf.app.run(launcher, sys.argv)
