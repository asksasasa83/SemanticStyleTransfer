from itertools import product
import neural_style
import os
import gc

CONTENT = 'examples/input/tubingen.jpg'
STYLE = 'examples/input/starry-night.jpg'

CONTENT_LAYERS = ['conv5_2', 'conv4_2', 'conv3_2', 'conv2_2']
STYLE_LAYERS = ['conv1_1,conv2_1,conv3_1,conv4_1,conv5_1', 
            'conv3_1,conv4_1,conv5_1', 
            'conv1_2,conv2_2,conv3_2,conv4_2,conv5_2',
            'conv1_2,conv2_2,conv3_4,conv4_4,conv5_4']
INIT=['random', 'image']
STYLE_WEIGHTS = [5, 100, 500, 2000]
OPTIMIZERS = ['Adam', 'L-BFGS']
NUM_ITERATIONS = 800
TV_WEIGHTS = [0.00001, 0.0001, 0.001]


def maybe_create_subfolders(folder):
    os.makedirs(folder, exist_ok=True)

if __name__ == '__main__':
    for cont, style, style_weight, optimizer, tv_weight, init in product(CONTENT_LAYERS, STYLE_LAYERS, STYLE_WEIGHTS, OPTIMIZERS, TV_WEIGHTS, INIT):
        folder_name = os.path.join('output', cont, style, str(style_weight), optimizer, init, str(tv_weight))

        maybe_create_subfolders(folder_name)

        filename = os.path.basename(CONTENT)[:-4] + '_' + os.path.basename(STYLE)[:-4] + '.jpg'
        try:
            neural_style.NeuralStyle(
                content_image=CONTENT,
                content_layers=cont,
                content_weight=5e0,
                image_size=500,
                init = init,
                num_iterations=NUM_ITERATIONS,
                output_image=os.path.join(folder_name, filename),
                print_iter=50,
                save_iter=50,
                style_image=STYLE,
                style_layers=style,
                style_weight=style_weight,
                tv_weight=tv_weight,
                optimizer=optimizer
            ).run()
        except:
            gc.collect()
            try:
                neural_style.NeuralStyle(
                    content_image=CONTENT,
                    content_layers=cont,
                    content_weight=5e0,
                    image_size=500,
                    init = init,
                    num_iterations=NUM_ITERATIONS,
                    output_image=os.path.join(folder_name, filename),
                    print_iter=50,
                    save_iter=50,
                    style_image=STYLE,
                    style_layers=style,
                    style_weight=style_weight,
                    tv_weight=tv_weight,
                    optimizer=optimizer
                ).run()
            except:
                pass
       