import os
import shutil

import v1.green_detector
import v2.shadow_corrected_green_detector
import v3.noise_reduction_green_detector
import optimize
import core.configurator as config
from core.configurator import load_string, get_default_configuration
from cut_houses import create_house_masks, cut_mask_from_image
from shadow.Shadow_Detection import shadow_detection, shadow_correction
from cv2 import imread, IMREAD_GRAYSCALE
import good_paramters as init

folder = "./../data/Munich/2023/raw/"
data = [
    (f'{folder}32688_5332.tif', f'{folder}32688_5332_label.png'),
    (f'{folder}32690_5335.tif', f'{folder}32690_5335_label.png'),
    (f'{folder}32691_5334.tif', f'{folder}32691_5334_label.png'),
    (f'{folder}32692_5335.tif', f'{folder}32692_5335_label.png'),
    (f'{folder}32692_5347.tif', f'{folder}32692_5347_label.png'),
]
population_size: int = 200
mutation_chance: float = 1.0 / 22.00
generations: int = 10000000
numer_of_results: int = 10


def run_optimize():
    work_dir = "./../temp/work_dir/"
    house_mask_name = "_house_mask.png"
    cut_label_name = "_cut_label.png"
    shadow_mask_name = "_shadow_mask.png"
    shadow_reduced_name = "_no_shadow.png"

    if os.path.isdir(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir)

    prepared_data = {}

    for image, label in data:
        print(f'Prepearing {image}...', end='', flush=True)
        name = image.split('/')[-1][:-len('.tif')]
        shutil.copy(image,  work_dir + name + '.tif')
        create_house_masks(image, work_dir + name + house_mask_name)
        cut_mask_from_image(label, work_dir + name + house_mask_name, work_dir + name + cut_label_name)

        shadow_detection(image, work_dir + name + shadow_mask_name)
        shadow_correction(image, work_dir + name + shadow_mask_name, work_dir + name + shadow_reduced_name)

        prepared_data[name] = {
            'image': imread(work_dir + name + '.tif'),
            'cut_label': imread(work_dir + name + cut_label_name, IMREAD_GRAYSCALE),
            'shadow_reduced': imread(work_dir + name + shadow_reduced_name),
            'house_mask': imread(work_dir + name + house_mask_name, IMREAD_GRAYSCALE),
        }
        print(f' Done.')

    best = optimize.run(prepared_data,
                        population_size=population_size,
                        mutation_chance=mutation_chance,
                        generations=generations,
                        numer_of_results=numer_of_results,
                        running_results=work_dir + "current_best.txt",
                        seed_parameters=[
                            #get_default_configuration(),
                            #get_default_configuration(),
                            #load_string(init.score_0_9407),
                            #load_string(init.score_0_9403),
                            #load_string(init.score_0_9404),
                            #load_string(init.score_0_9405),
                            #load_string(init.score_0_94052),
                        ])

    for i, config in enumerate(best):
        print(f'Position {i+1}:\n{config}\n')


if __name__ == '__main__':
    # v1.green_detector.process_image("00_input.tif")
    # v2.shadow_corrected_green_detector.process_image("00_input.tif")
    # v3.noise_reduction_green_detector.process_image("00_input.tif")

    # default = config.get_default_configuration()
    # defaultRunner = config.create_process(default)
    # res = defaultRunner(f'{folder}00_input.png', None)

    run_optimize()


