"""Parent selection probability distributions over a population."""

from abc import ABC, abstractmethod

from auxein.population import Item, Population


class Distribution(ABC):
    @abstractmethod
    def get(self, population: Population) -> list[tuple[str, float]]:
        pass


class Fps(Distribution):
    def __init__(self) -> None:
        super().__init__()

    def get(self, population: Population) -> list[tuple[str, float]]:
        if any(item.fitness < 0 for item in population.pool):
            raise ValueError("Fps requires non-negative fitness values: use FpsWithWindowing or SigmaScaling for negative fitness.")
        total_fitness = population.total_fitness()
        if total_fitness == 0:
            raise ValueError("Fps requires a strictly positive total fitness: use FpsWithWindowing or SigmaScaling.")
        return list(map(lambda item: (item.individual.id, item.fitness / total_fitness), population.pool))


def _uniform(population: Population) -> list[tuple[str, float]]:
    probability = 1 / population.size()
    return [(item.individual.id, probability) for item in population.pool]


class FpsWithWindowing(Distribution):
    def __init__(self) -> None:
        super().__init__()

    def __scale_fitness_function(self, item: Item, minimum_fitness: float) -> float:
        return item.fitness - minimum_fitness

    def get(self, population: Population) -> list[tuple[str, float]]:
        minimum_fitness = population.min_fitness()
        total_fitness = sum(self.__scale_fitness_function(item, minimum_fitness) for item in population.pool)
        if total_fitness == 0:
            return _uniform(population)
        return list(
            map(lambda item: (item.individual.id, self.__scale_fitness_function(item, minimum_fitness) / total_fitness), population.pool)
        )


class SigmaScaling(Distribution):
    def __init__(self) -> None:
        super().__init__()

    def get(self, population: Population) -> list[tuple[str, float]]:
        if population.min_fitness() == population.max_fitness():
            return _uniform(population)
        lower_bound = population.mean_fitness() - 2 * population.std_fitness()
        scaled = [(item.individual.id, max(item.fitness - lower_bound, 0)) for item in population.pool]
        total_fitness = sum(value for _, value in scaled)
        if total_fitness == 0:
            return _uniform(population)
        return [(individual_id, value / total_fitness) for individual_id, value in scaled]
