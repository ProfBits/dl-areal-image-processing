import v1.green_detector
import v2.shadow_corrected_green_detector
import v3.noise_reduction_green_detector
import optimize

folder = "image-processing/testing/"
data = [
    (f'{folder}00_input.png', f'{folder}00_input.png'),
    (f'{folder}32692_5347.png', f'{folder}32692_5347_corrected.png')
]
population_size: int = 50
mutation_chance: float = 0.025
generations: int = 10
numer_of_results: int = 5


if __name__ == '__main__':
    #v1.green_detector.process_image("00_input.tif")
    #v2.shadow_corrected_green_detector.process_image("00_input.tif")
    #v3.noise_reduction_green_detector.process_image("00_input.tif")

    best = optimize.run(data,
                 population_size=population_size,
                 mutation_chance=mutation_chance,
                 generations=generations,
                 numer_of_results=numer_of_results)

    for i, config in enumerate(best):
        print(f'Position {i+1}:\n{config}\n')

