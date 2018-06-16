import subprocess
import glob
import os

PATH_NEURAL_GRAM = '../deep-painterly-harmonization/neural_gram.py'

def gatys(c_path, s_path, m_path, md_path, o_path, i):
    subprocess.check_output(['python', 'neural_style.py', 
            '--content_image', c_path, 
            '--init', 'image', 
            '--optimizer', 'Adam', 
            '--num_iterations', '1000', 
            '--style_image', s_path, 
            '--style_weight', '500', 
            '--hist_weight', '0'], 
                stderr=subprocess.STDOUT)

def gatys_histogram(c_path, s_path, m_path, md_path, o_path, i):
    subprocess.check_output(['python', 'neural_style.py', 
            '--content_image', c_path, 
            '--init', 'image', 
            '--optimizer', 'Adam', 
            '--num_iterations', '1000', 
            '--style_image', s_path, 
            '--style_weight', '500', 
            '--hist_weight', '0.01'], 
                stderr=subprocess.STDOUT)

def dpa1(c_path, s_path, m_path, md_path, o_path, i):
    subprocess.check_output(['th', 'neural_gram.lua',
            '-content_image', c_path,
            '-style_image', s_path,
            '-tmask_image', m_path,
            '-mask_image', md_path,
            '-gpu', '0', '-original_colors', '0', '-image_size', '700',
            '-output_image', o_path,
            '-print_iter', '100', '-save_iter', '100'])

def dpa2(c_path, s_path, m_path, md_path, o_path, i):
    o_path_interres = '.'.join(o_path.split('.')[:-1]) + 'inter_res.jpg
    subprocess.check_output(['th', 'neural_gram.lua',
            '-content_image', c_path,
            '-style_image', s_path,
            '-tmask_image', m_path,
            '-mask_image', md_path,
            '-gpu', '0', '-original_colors', '0', '-image_size', '700',
            '-output_image', o_path_interres,
            '-print_iter', '100', '-save_iter', '100'])
    
    subprocess.check_output(['th', 'neural_paint.lua',
            '-content_image', c_path,
            '-style_image', s_path,
            '-tmask_image', m_path,
            '-mask_image', md_path,
            '-cnnmrf_image', o_path_interres,
            '-gpu', '0', '-original_colors', '0', '-image_size', '700',
            '-index', str(i), '-wikiart_fn', 'wikiart_output.txt',
            '-output_image', o_path,
            '-print_iter', '100', '-save_iter', '100',
            '-num_iterations', '1000'])

def photographic(c_path, s_path, m_path, md_path, o_path, i):
    subprocess.check_output(['python', 'deep_photostyle.py', 
        '--content_image_path', c_path,
        '--style_image_path', s_path,
        '--content_seg_path', m_path,
        '--style_seg_path', m_path,
        '--output_image', o_path,
        '--max_iter', '700',
        '--style_option', '2',
        '--serial', os.path.dirname(o_path)])

algorithms = [gatys, gatys_histogram, dpa1, dpa2, photographic]
algo_names = ['gatys', 'gatys_histogram', 'dpa1', 'dpa2', 'photo']

def transfer_folder(folder):
    c_path = os.path.join(folder, 'content.jpg')
    s_path = os.path.join(folder, 'style.jpg')
    m_path = os.path.join(folder, 'mask.jpg')
    md_path = os.path.join(folder, 'mask_dilated.jpg')
    
    for i, algo,name in enumerate(zip(algorithms, algo_names)):
        o_path = os.path.join(folder, name, 'output.jpg')
        try:
            print("Computing algorithm {} of folder {}".format(i, folder))
            algo(c_path, s_path, m_path, md_path, o_path, i)
        except Exception as e:
            print("An error occurred, {}".format(e))

def main():
    for folder in glob.glob(os.path.join('DATASET', '*/')):
        transfer_folder(folder)

        
    
if __name__=='__main__':
    transfer_folder(os.path.join('DATASET', '1'))