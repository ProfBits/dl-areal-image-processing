from random import choice

import numpy as np
import cv2

import parameterized.configurator as config
from mask_evaluation import evaluate_masks


def __evaluate(process, data) -> float:
    res = [(mask, process(image)) for image, mask in data]
    evaluation = evaluate_masks(res)
    return evaluation.F1


def run(data: list[tuple[str, str]],
        population_size: int = 50,
        mutation_chance: float = 0.025,
        generations: int = 10_000_000,
        numer_of_results: int = 5) -> list[str]:
    """
    Optimized the parameter for the image processing approach with a genetic algorithm
    Args:
        data: the sample data to train with. Tuples of input and expected mask
        population_size: the number of active configurations
        mutation_chance: the chance that a parameter chooses an independent random value on crossover
        generations: the number of generations after with to stop
        numer_of_results: the number of best configurations to return

    Returns:
        list[str]: The best n configurations serialized to a string
    """

    data: list[tuple[str, cv2.typing.MatLike]] = [(image, cv2.imread(mask, cv2.IMREAD_GRAYSCALE)) for image, mask in data]
    population = [config.get_random_configuration() for _ in range(population_size)]
    population[0] = config.get_default_configuration()
    population[1] = config.get_default_configuration()
    results = []
    for generation in range(generations):
        print(f'Generation {generation}: Starting...', end='\r', flush=True)
        processes = [config.create_process(parameters) for parameters in population]
        print(f'Generation {generation}: Running...', end='\r', flush=True)
        results = [__evaluate(process, data) for process in processes]

        best = np.max(results)
        bar = max(np.median(results), 0.1)
        print(f'Generation {generation}: Best: {best:2.3f}, Requirement: {bar:2.3f} | Reproducing...', end='\r', flush=True)

        with open("Results.txt", 'w') as fp:
            winners = list(zip(population, results))
            winners.sort(key=lambda t: t[1], reverse=True)
            winners = winners[:numer_of_results]
            for winner in winners:
                fp.write(f"Score: {winner[1]}\n")
                fp.write(config.to_string(winner[0]))
                fp.write("\n\n")

        survivors = [parameters for parameters, score in zip(population, results) if score >= bar]
        survivor_count = len(survivors)
        population = [e for e in survivors]
        for _ in range(population_size - survivor_count):
            population.append(config.cross_over(choice(survivors), choice(survivors), mutation_chance))

        print(f'Generation {generation}: Best: {np.max(results):2.3f}, Requirement: {bar:2.3f} | Done.')

    winners = list(zip(population, results))
    winners.sort(key=lambda t: t[1], reverse=True)
    return winners[:numer_of_results]

