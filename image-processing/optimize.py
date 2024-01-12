import json
from random import choice
from typing import Optional
from datetime import datetime as dt

import numpy as np
import cv2

import core.configurator as config

import sys
sys.path.append("./../")
from mask_evaluation import evaluate_masks


tstamp_format = "%d/%m/%Y %H:%M:%S"


def __evaluate(process, data) -> float:
    res = [(mask, process(image)) for image, mask in data]
    evaluation = evaluate_masks(res)
    return evaluation.F1


def run(data: list[tuple[str, str]] | dict[str, dict[str, np.ndarray]],
        population_size: int = 50,
        mutation_chance: float = 0.025,
        generations: int = 10_000_000,
        numer_of_results: int = 5,
        running_results: Optional[str] = None,
        seed_parameters: Optional[list] = None,
        ) -> list[str]:
    """
    Optimized the parameter for the image processing approach with a genetic algorithm
    Args:
        data: the sample data to train with. Tuples of input and expected mask
        population_size: the number of active configurations
        mutation_chance: the chance that a parameter chooses an independent random value on crossover
        generations: the number of generations after with to stop
        numer_of_results: the number of best configurations to return
        running_results: an optional file to write the best configurations to each step
        seed_parameters: a list of parameters to add to the initial population
    Returns:
        list[str]: The best n configurations serialized to a string
    """

    if isinstance(data, list):
        data: list[tuple[str, np.ndarray]] = [
            (image, cv2.imread(mask, cv2.IMREAD_GRAYSCALE)) for image, mask in data]
    else:
        data: list[tuple[dict[str, np.ndarray], np.ndarray]] = [(val, val['cut_label']) for val in data.values()]
    new_population = [config.get_random_configuration()
                  for _ in range(population_size)]

    if seed_parameters is None:
        new_population[0] = config.get_default_configuration()
        new_population[1] = config.get_default_configuration()
    else:
        for i, val in enumerate(seed_parameters):
            new_population[i] = val

    population = []
    results = []
    survivors = []
    survivor_results = []
    for generation in range(generations):
        print(f'{dt.now().strftime(tstamp_format)} Generation {generation}: Starting...', end='\r', flush=True)
        processes = [config.create_process(parameters)
                     for parameters in new_population]
        print(f'{dt.now().strftime(tstamp_format)} Generation {generation}: Running...', end='\r', flush=True)
        new_results = [__evaluate(process, data) for process in processes]

        # combine results
        results = survivor_results + new_results
        population = survivors + new_population

        best = np.max(results)
        bar = max(np.median(results), 0.1)
        print(
            f'{dt.now().strftime(tstamp_format)} Generation {generation}: Best: {best:2.3f}, Requirement: {bar:2.3f} | Reproducing...', end='\r', flush=True)

        if running_results is not None:
            winners = list(zip(population, results))
            winners.sort(key=lambda t: t[1], reverse=True)
            winners = winners[:numer_of_results]

            with open(running_results, 'w') as fp:
                for winner in winners:
                    fp.write(f"Score: {winner[1]}\n")
                    fp.write(config.to_string(winner[0]))
                    fp.write("\n\n")

        survivor_stats = [(parameters, score) for parameters, score in zip(
            population, results) if score >= bar]
        survivors = [parameters for parameters, _ in survivor_stats]
        survivor_results = [score for _, score in survivor_stats]

        survivor_count = len(survivors)
        new_population = [config.cross_over(choice(survivors), choice(survivors), mutation_chance)
                          for _ in range(population_size - survivor_count)]

        print(
            f'{dt.now().strftime(tstamp_format)} Generation {generation}: Best: {np.max(results):2.3f}, Requirement: {bar:2.3f} | Done.')

    winners = list(zip(population, results))
    winners.sort(key=lambda t: t[1], reverse=True)
    return winners[:numer_of_results]
