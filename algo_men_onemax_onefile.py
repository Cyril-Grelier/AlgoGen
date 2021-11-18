import dataclasses
import math
import random
import statistics
import textwrap
from abc import ABC, abstractmethod
from copy import copy, deepcopy
from dataclasses import InitVar, field
from random import randint
from statistics import mean
from typing import List, Tuple

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


size_indiv = 100


import dataclasses
from enum import Enum
from typing import List


class Selection_type(Enum):
    select_random: int = 1
    select_best: int = 2
    select_tournament: int = 3
    select_wheel: int = 4


class Crossover_type(Enum):
    mono_point: int = 1
    multipoint: int = 2
    uniforme: int = 3


class Mutation_type(Enum):
    flip_1: int = 1
    flip_3: int = 2
    flip_5: int = 3
    bit_flip: int = 4


class Insertion_type(Enum):
    age: int = 1
    fitness: int = 2


class Adaptive_type(Enum):
    fixed_roulette_wheel: int = 1
    adaptive_roulette_wheel: int = 2
    adaptive_pursuit: int = 3
    ucb: int = 4


@dataclasses.dataclass
class Parameters:
    """dataclass for parameters"""

    nb_max_iterations: int

    population_size: int
    selection_type: Selection_type
    proportion_selection: float
    crossover: List[Crossover_type]
    adaptive_crossover: Adaptive_type
    proportion_crossover: float
    mutation: List[Mutation_type]
    adaptive_mutation: Adaptive_type
    proportion_mutation: float
    insertion: Insertion_type
    p_min: float = 0.15
    beta: float = 1.5


class IndividualOneMax:
    def __init__(self):
        self.sequence = [0] * size_indiv  # size indiv
        self.fitness = 0
        self.age = 0

    def compute_fitness(self):
        self.fitness = sum(self.sequence)
        return self.fitness

    def __eq__(self, other):
        return self.sequence == other.sequence

    def __repr__(self):
        return f"{self.fitness} {id(self)} {''.join([str(i) for i in self.sequence])}"

    def __hash__(self):
        r = "".join([str(i) for i in self.sequence])
        return int(r, 2)

    def __getitem__(self, key):
        return self.sequence[key]

    def __setitem__(self, key, value):
        self.sequence[key] = value
        return value


@dataclasses.dataclass
class Stats:

    max_fitness: List[int] = field(default_factory=list)
    min_fitness: List[int] = field(default_factory=list)
    mean_fitness: List[float] = field(default_factory=list)
    fitness_diversity: List[int] = field(default_factory=list)
    total_fitness: List[int] = field(default_factory=list)
    diversity: List[int] = field(default_factory=list)
    max_age: List[int] = field(default_factory=list)
    mean_age: List[float] = field(default_factory=list)
    utility_crossover: List[int] = field(default_factory=list)
    proba_crossover: List[float] = field(default_factory=list)
    past_crossover: List[Crossover_type] = field(default_factory=list)
    utility_mutation: List[int] = field(default_factory=list)
    proba_mutation: List[float] = field(default_factory=list)
    past_mutation: List[Mutation_type] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"{self.max_fitness=}\n"
            f"{self.min_fitness=}\n"
            f"{self.mean_fitness=}\n"
            f"{self.fitness_diversity=}\n"
            f"{self.diversity=}\n"
            f"{self.max_age=}\n"
            f"{self.mean_age=}\n"
            f"{self.utility_crossover=}\n"
            f"{self.proba_crossover=}\n"
            f"{self.past_crossover=}\n"
            f"{self.utility_mutation=}\n"
            f"{self.proba_mutation=}\n"
            f"{self.past_mutation=}\n"
        )


class Algo_genetic:
    def __init__(self, parameters: Parameters) -> None:
        self.parameters: Parameters = parameters
        self.population: list(IndividualOneMax) = [
            IndividualOneMax() for _ in range(parameters.population_size)
        ]
        self.stats: Stats = Stats()
        self.stats.proba_crossover = [1 / len(parameters.crossover)] * len(
            parameters.crossover
        )
        self.stats.proba_mutation = [1 / len(parameters.mutation)] * len(
            parameters.mutation
        )
        self.selected: list(IndividualOneMax) = []
        self.nb_select: int = int(
            parameters.population_size * parameters.proportion_selection
        )
        self.crossed: list(IndividualOneMax) = []
        self.mutated: list(IndividualOneMax) = []

        self.nb_turn: int = 0

    def sort_pop_fitness(self):
        self.population.sort(key=lambda i: i.fitness, reverse=True)

    def sort_pop_age(self):
        self.population.sort(key=lambda i: i.age, reverse=False)

    def final_condition(self):
        return (
            self.stats.max_fitness[-1] == size_indiv  # size indiv
            or self.nb_turn >= self.parameters.nb_max_iterations
        )

    def run(self):
        self.statistic()
        while not self.final_condition():
            self.nb_turn += 1
            # population get older
            for indiv in self.population:
                indiv.age += 1
            self.sort_pop_fitness()
            self.selection()
            self.crossover()
            self.mutation()
            self.insertion()
            self.statistic()

    def selection(self):
        self.selected = []
        if self.parameters.selection_type == Selection_type.select_best:
            self.selected = self.population[0 : self.nb_select]
        elif self.parameters.selection_type == Selection_type.select_random:
            self.selected = random.sample(self.individuals, self.nb_select)
        elif self.parameters.selection_type == Selection_type.select_tournament:
            self.select_tournament()
        elif self.parameters.selection_type == Selection_type.select_wheel:
            self.select_wheel()

    def select_tournament(self):
        nb_winners = 2
        nb_players = 5
        while len(self.selected) < self.nb_select:
            selected = sorted(
                random.sample(self.population, nb_players),
                key=lambda i: i.fitness,
                reverse=True,
            )[0:nb_winners]
            self.selected.extend(selected)
        self.selected = self.selected[0 : self.nb_select]

    def select_wheel(self):
        fits = [individual.fitness for individual in self.population]
        total = sum(fits)
        wheel = [1] * len(fits) if total == 0 else [f / total for f in fits]
        probabilities = [sum(wheel[: i + 1]) for i in range(len(wheel))]
        for _ in range(self.nb_select):
            r = random.random()
            for i, individual in enumerate(self.population):
                if r <= probabilities[i]:
                    self.selected.append(deepcopy(individual))
                    break

    def adaptive(self, is_crossover: bool) -> Crossover_type:
        adaptive_type = (
            self.parameters.adaptive_crossover
            if is_crossover
            else self.parameters.adaptive_mutation
        )
        operators = (
            self.parameters.crossover if is_crossover else self.parameters.mutation
        )
        proba_operator = (
            self.stats.proba_crossover if is_crossover else self.stats.proba_mutation
        )

        if adaptive_type == Adaptive_type.fixed_roulette_wheel:
            return random.choices(operators, proba_operator, k=1)[0]

        utility = (
            self.stats.utility_crossover
            if is_crossover
            else self.stats.utility_mutation
        )

        past_operators = (
            self.stats.past_crossover if is_crossover else self.stats.past_mutation
        )

        if adaptive_type == Adaptive_type.adaptive_roulette_wheel:
            pmin = 0.05
            utility = utility[-20:]
            past_operators = past_operators[-20:]
            n = len(operators)
            if len(utility) >= 10:
                sum_uk = sum(utility)
                for i in range(n):
                    m = operators[i]
                    u_n = sum([u for u, me in zip(utility, past_operators) if m == me])
                    proba_operator[i] = pmin + (1 - n * pmin) * (
                        u_n / sum_uk if sum_uk != 0 else 1
                    )
            return random.choices(operators, proba_operator, k=1)[0]

        if adaptive_type == Adaptive_type.adaptive_pursuit:
            if len(utility) % 10 == 0:
                # reset utility TODO check if it works
                for i in range(len(operators)):
                    proba_operator[i] = 1 / len(operators)
            utility = utility[-20:]
            past_operators = past_operators[-20:]
            if len(utility) >= 10:
                pmin = 0.05
                pmax = 1 - (len(operators) - 1) * pmin
                beta = 1.5
                utilities = [
                    sum([u for u, op in zip(utility, past_operators) if op == operator])
                    for operator in operators
                ]
                maxi = max(utilities)
                indices = [i for i, v in enumerate(utilities) if v == maxi]
                best = random.choice(indices)
                for i in range(len(operators)):
                    tminus1 = proba_operator[i]
                    if i == best:
                        proba_operator[i] = tminus1 + beta * (pmax - tminus1)
                    else:
                        proba_operator[i] = tminus1 + beta * (pmin - tminus1)
            return random.choices(operators, proba_operator, k=1)[0]

        if adaptive_type == Adaptive_type.ucb:
            if len(utility) >= 10:
                exploitations = []
                explorations = []
                for i in range(len(operators)):
                    methode = operators[i]
                    nb_i = [m for m in past_operators].count(methode)
                    exploitations.append(
                        sum(
                            [
                                u
                                for u, me in zip(utility, past_operators)
                                if methode == me
                            ]
                        )
                    )
                    explorations.append(
                        math.sqrt(2 * math.log(len(past_operators)) / (nb_i + 1))
                    )
                for i in range(len(operators)):
                    proba_operator[i] = exploitations[i] + explorations[i]
            maxi = max(proba_operator)
            indices = [
                op for op, proba in zip(operators, proba_operator) if proba == maxi
            ]
            chosen = random.choice(indices)
            return chosen

    def crossover(self):
        self.crossed = []
        operator: Crossover_type = self.adaptive(is_crossover=True)

        if operator == Crossover_type.mono_point:
            for i in range(0, len(self.selected), 2):
                first_child = IndividualOneMax()
                second_child = IndividualOneMax()
                rand_cross = random.random()
                if rand_cross < self.parameters.proportion_crossover:
                    i1 = self.selected[i]
                    i2 = self.selected[i + 1]
                    rand = random.randint(1, len(i1.sequence))
                    first_child[::] = copy(i1[0:rand]) + copy(i2[rand:])
                    second_child[::] = copy(i2[0:rand]) + copy(i1[rand:])
                else:
                    first_child.sequence = copy(self.selected[i].sequence)
                    second_child.sequence = copy(self.selected[i + 1].sequence)
                self.crossed.extend([first_child, second_child])

        elif operator == Crossover_type.multipoint:
            for i in range(0, len(self.selected), 2):
                first_child = IndividualOneMax()
                second_child = IndividualOneMax()
                rand_cross = random.random()
                if rand_cross < self.parameters.proportion_crossover:
                    i1 = self.selected[i]
                    i2 = self.selected[i + 1]
                    rand1 = random.randint(1, len(i1.sequence) - 2)
                    rand2 = random.randint(rand1, len(i1.sequence))
                    first_child[::] = (
                        deepcopy(i1[0:rand1])
                        + deepcopy(i2[rand1:rand2])
                        + deepcopy(i1[rand2:])
                    )
                    second_child[::] = (
                        deepcopy(i2[0:rand1])
                        + deepcopy(i1[rand1:rand2])
                        + deepcopy(i2[rand2:])
                    )
                else:
                    first_child.sequence = deepcopy(self.selected[i][::])
                    second_child.sequence = deepcopy(self.selected[i + 1].sequence)
                self.crossed.extend([first_child, second_child])

        elif operator == Crossover_type.uniforme:
            for i in range(0, len(self.selected), 2):
                first_child = IndividualOneMax()
                second_child = IndividualOneMax()
                rand_cross = random.random()
                if rand_cross < self.parameters.proportion_crossover:
                    i1 = self.selected[i]
                    i2 = self.selected[i + 1]
                    for j in range(size_indiv):  # size indiv
                        first_child[j], second_child[j] = (
                            (i1[j], i2[j]) if random.random() <= 0.5 else (i2[j], i1[j])
                        )
                else:
                    first_child.sequence = deepcopy(self.selected[i][::])
                    second_child.sequence = deepcopy(self.selected[i + 1].sequence)
                self.crossed.extend([first_child, second_child])

        self.stats.past_crossover.append(operator)
        self.stats.utility_crossover.append(
            statistics.mean([indiv.compute_fitness() for indiv in self.crossed])
        )

    def mutation(self):
        self.mutated = []
        operator: Crossover_type = self.adaptive(is_crossover=False)

        if self.parameters.mutation[0] == Mutation_type.bit_flip:
            for indiv in self.crossed:
                if random.random() <= self.parameters.proportion_mutation:
                    length = size_indiv  # size indiv
                    p = 1 / length
                    for i in range(length):
                        if random.random() <= p:
                            indiv.sequence[i] = 1 - indiv.sequence[i]
                self.mutated.append(indiv)
        elif self.parameters.mutation[0] == Mutation_type.flip_1:
            for indiv in self.crossed:
                if random.random() <= self.parameters.proportion_mutation:
                    for i in random.sample(range(len(indiv.sequence)), 1):
                        indiv.sequence[i] = 1 - indiv.sequence[i]
                self.mutated.append(indiv)
        elif self.parameters.mutation[0] == Mutation_type.flip_3:
            for indiv in self.crossed:
                if random.random() <= self.parameters.proportion_mutation:
                    for i in random.sample(range(len(indiv.sequence)), 3):
                        indiv.sequence[i] = 1 - indiv.sequence[i]
                self.mutated.append(indiv)
        elif self.parameters.mutation[0] == Mutation_type.flip_5:
            for indiv in self.crossed:
                if random.random() <= self.parameters.proportion_mutation:
                    for i in random.sample(range(len(indiv.sequence)), 5):
                        indiv.sequence[i] = 1 - indiv.sequence[i]
                self.mutated.append(indiv)

        self.stats.past_mutation.append(operator)
        self.stats.utility_mutation.append(
            statistics.mean([indiv.compute_fitness() for indiv in self.mutated])
        )

    def insertion(self):
        if self.parameters.insertion == Insertion_type.fitness:
            self.sort_pop_fitness()
        elif self.parameters.insertion == Insertion_type.age:
            self.sort_pop_age()

        self.population[-len(self.mutated) :] = self.mutated

        selection = statistics.mean([indiv.fitness for indiv in self.selected])
        crossover = self.stats.utility_crossover[-1]
        mutation = self.stats.utility_mutation[-1]
        u_crossover = (crossover - selection) / size_indiv  # size indiv
        u_mutation = (mutation - crossover) / size_indiv  # size indiv
        self.stats.utility_crossover[-1] = u_crossover
        self.stats.utility_mutation[-1] = u_mutation

    def statistic(self):
        self.sort_pop_fitness()
        fits = [individual.fitness for individual in self.population]
        ages = [individual.age for individual in self.population]
        self.stats.total_fitness.append(sum(fits))
        self.stats.max_fitness.append(fits[0])
        self.stats.min_fitness.append(fits[-1])
        self.stats.mean_fitness.append(statistics.mean(fits))
        self.stats.fitness_diversity.append(len(set(self.population)))
        self.stats.diversity.append(len(set(self.population)))
        self.stats.max_age.append(max(ages))
        self.stats.mean_age.append(statistics.mean(ages))


def annot_max(x, y, ax=None):
    xmax = x[np.argmax(y)]
    ymax = y.max()
    text = f"x={xmax}, y={ymax}"
    if not ax:
        ax = plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops = dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(
        xycoords="data",
        textcoords="axes fraction",
        arrowprops=arrowprops,
        bbox=bbox_props,
        ha="right",
        va="top",
    )
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.94, 0.96), **kw)


def show_stats(stats: Stats, parameters: Parameters, title=None):
    fig, ax = plt.subplots(figsize=(8, 8))

    if title:
        plt.title(title)
    plt.xlabel("tours")
    plt.ylabel("fitness")
    plt.xlim(0, len(stats.max_fitness))
    plt.ylim(0, 100)
    # (-70,0)#(0, 100)
    # stats['parameters']['chromosome size'])
    # stats['max_fitness'][-1]*1.1

    ax.plot(stats.max_fitness, color="red", label="max")
    ax.plot(stats.min_fitness, color="green", label="min")
    ax.plot(stats.mean_fitness, color="blue", label="mean")
    ax.plot(stats.fitness_diversity, color="black", label="fitness_diversity")

    # windows_size = 49
    # polynomial_order = 3

    # ax.plot(savgol_filter(stats['max_fitness'], windows_size, polynomial_order), color='red', linestyle='dashed',
    #         label='max_fitness soft')
    # ax.plot(savgol_filter(stats['min_fitness'], windows_size, polynomial_order), color='green', linestyle='dashed',
    #         label='min_fitness soft')
    # ax.plot(savgol_filter(stats['mean_fitness'], windows_size, polynomial_order), color='blue', linestyle='dashed',
    #         label='mean_fitness soft')

    plt.legend(title="fitness", loc="lower right")

    plt.title("\n".join(textwrap.wrap(str(parameters.mutation), 120)))
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.plot(stats.diversity, color="yellow", label="diversity")
    ax.plot(stats.max_age, color="c", label="max_age")
    ax.plot(stats.mean_age, color="m", label="mean_age")

    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)

    plt.show()


# def main():

#     p = Parameters(
#         nb_max_iterations=5000,
#         population_size=100,
#         selection_type=Selection_type.select_best,
#         proportion_selection=0.4,
#         crossover=[Crossover_type.uniforme],
#         adaptive_crossover=Adaptive_type.fixed_roulette_wheel,
#         proportion_crossover=1,
#         mutation=[
#             Mutation_type.flip_1,
#             Mutation_type.flip_3,
#             Mutation_type.flip_5,
#             Mutation_type.bit_flip,
#         ],
#         adaptive_mutation=Adaptive_type.ucb,
#         # Adaptive_type.fixed_roulette_wheel,
#         proportion_mutation=1,
#         insertion=Insertion_type.fitness,
#     )
#     population = Algo_genetic(p)
#     population.run()

#     print(population.stats)
#     print(population.nb_turn)
#     show_stats(population.stats, population.parameters)


def main():
    p = Parameters(
        nb_max_iterations=5000,
        population_size=100,
        selection_type=Selection_type.select_tournament,
        proportion_selection=0.02,
        crossover=[Crossover_type.uniforme],
        adaptive_crossover=Adaptive_type.fixed_roulette_wheel,
        proportion_crossover=1,
        mutation=[
            Mutation_type.flip_1,
            Mutation_type.flip_3,
            Mutation_type.flip_5,
            Mutation_type.bit_flip,
        ],
        adaptive_mutation=Adaptive_type.ucb,
        proportion_mutation=1,
        insertion=Insertion_type.fitness,
    )

    diff_adaptive_mutation = [
        Adaptive_type.fixed_roulette_wheel,
        Adaptive_type.adaptive_roulette_wheel,
        Adaptive_type.adaptive_pursuit,
        Adaptive_type.ucb,
    ]

    mutations = [
        Mutation_type.flip_1,
        Mutation_type.flip_3,
        Mutation_type.flip_5,
        Mutation_type.bit_flip,
    ]

    for mu in diff_adaptive_mutation:
        p.adaptive_mutation = mu
        population = Algo_genetic(p)
        population.run()
        print(mu)

        count = {m: [0] for m in mutations}
        for v in population.stats.past_mutation:
            for m in mutations:
                count[m].append(count[m][-1] + (1 if v == m else 0))

        fig, ax = plt.subplots(figsize=(10, 10))

        for m in count.keys():
            ax.plot(list(range(len(count[m]))), count[m], label=m)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)

        plt.title(mu)
        plt.show()

        show_stats(population.stats, p)
