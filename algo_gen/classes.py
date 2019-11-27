import copy
import math
import random
import statistics
from abc import ABC, abstractmethod

from algo_gen.tools.tools import get_import


###############################################################
#                           Gene                              #
###############################################################

class Gene(ABC):

    def __init__(self):
        self.bit = None

    @abstractmethod
    def mutate(self):
        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass


###############################################################
#                       Individual                            #
###############################################################


class Individual(ABC):

    def __init__(self, parameters):
        self.sequence = []
        self.parameters = parameters

    def __getitem__(self, key):
        return self.sequence[key]

    def __setitem__(self, key, value):
        self.sequence[key] = value
        return value

    @abstractmethod
    def fitness(self):
        pass

    @abstractmethod
    def __eq__(self, other):
        pass

    def __str__(self):
        return repr(self)

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def __hash__(self):
        pass


###############################################################
#                       Population                            #
###############################################################


class Population:

    def __init__(self, parameters):
        self.individuals = []
        self.parameters = parameters
        self.stats = {
            'max_fitness': [], 'min_fitness': [], 'mean_fitness': [], 'fitness_diversity': [], "total_fitness": [],
            'diversity': [], 'max_age': [], 'mean_age': [], 'parameters': parameters, 'time': 0.0, 'utility': []
        }
        self.selected = []
        self.crossed = []
        self.mutated = []
        self.nb_turns = 0
        self.individual_class = get_import(parameters, 'individual')

        individuals = []
        for _ in range(self.parameters['population size']):
            individuals.append(self.individual_class(parameters))

        fits = [i.fitness() for i in individuals]
        self.individuals = [(i, f, 0) for i, f in sorted(zip(individuals, fits), key=lambda z: z[1])]

        self.nb_select = int(len(self.individuals) * self.parameters['proportion selection'])
        description = f'Population parameters :' \
                      f'\n\tIndividuals : {self.individual_class}' \
                      f'\n\tSize of an individual : {self.parameters["chromosome size"]}' \
                      f'\n\tSize of the population : {self.parameters["population size"]}' \
                      f'\n\tNumber of individuals selected each turns : {self.nb_select}' \
                      f'\n\tSelection : {self.parameters["selection"]}' \
                      f'\n\tCrossover : {self.parameters["crossover"]} (' \
                      f'{self.parameters["proportion crossover"] * 100}%)' \
                      f'\n\tMutation : {self.parameters["mutation"]} (' \
                      f'{self.parameters["proportion mutation"] * 100}%)' \
                      f'\n\tInsertion : {self.parameters["insertion"]}'
        print(description)

    def sort_individuals_fitness(self):
        self.individuals.sort(key=lambda i: i[1], reverse=True)

    def sort_individuals_age(self):
        self.individuals.sort(key=lambda i: i[2], reverse=False)

    def population_get_older(self):
        self.nb_turns += 1
        self.individuals = [(i, f, (a + 1)) for i, f, a in self.individuals]

    def __repr__(self):
        s = f"Population size : {len(self.individuals)}"
        for i in self.individuals:
            s += "\n" + str(i)
        return s

    def final_condition(self):
        if self.nb_turns >= self.parameters['stop after no change']:
            last_max = self.stats['max_fitness'][-self.parameters['stop after no change']:]
            last_min = self.stats['min_fitness'][-self.parameters['stop after no change']:]
            max_change = not all(x >= y for x, y in zip(last_max, last_max[1:]))
            min_change = not all(x >= y for x, y in zip(last_min, last_min[1:]))
            return (not (self.nb_turns == self.parameters['nb turn max'])) and not (not max_change and not min_change)
        else:
            return not (self.nb_turns == self.parameters['nb turn max'])

    def start(self):
        self.statistic()
        if 'function_each_turn' in self.parameters:
            self.parameters['function_each_turn'](self)
        while self.final_condition():
            self.population_get_older()
            self.sort_individuals_fitness()
            self.turn()
            self.statistic()
        if 'function_end' in self.parameters:
            self.parameters['function_end'](self)

    def turn(self):
        self.selection()
        self.crossover()
        self.mutation()
        self.insertion()

    ###############################################################
    #                       Selection                             #
    ###############################################################

    def selection(self):
        """
        chose the type of selection to use and select individuals
        :return:
        """
        self.selected = []
        switch = {
            'select_random': self.select_random,
            'select_best': self.select_best,
            'select_tournament': self.select_tournament,
            'select_wheel': self.select_wheel,
            'adaptative': self.select_adaptative,
        }
        switch[self.parameters['selection'][0]]()
        self.stats['utility'][-1][0][0] = statistics.mean(self.stats['utility'][-1][0][0])

    def select_adaptative(self):
        switch = {
            'fixed roulette wheel': self.select_adaptative_fixe,
            'adaptive roulette wheel': self.select_adaptative_wheel,
            'adaptive pursuit': self.select_adaptative_poursuit,
            'UCB': self.select_adaptative_ucb,
            'DMAB': self.select_adaptative_dmab,
        }
        switch[self.parameters['selection'][1]]()

    def select_adaptative_fixe(self):
        proba, select = list(map(list, zip(*self.parameters['selection'][2])))
        selection = random.choices(range(len(select)), proba, k=1)[0]
        switch = {
            'select_random': self.select_random,
            'select_best': self.select_best,
            'select_tournament': self.select_tournament,
            'select_wheel': self.select_wheel,
        }
        if self.parameters['selection'][2][selection][1][0] == 'select_tournament':
            self.select_tournament(self.parameters['selection'][2][selection][1])
        else:
            switch[self.parameters['selection'][2][selection][1][0]]()
        print(self.parameters['selection'][2][selection][1][0])

    def select_adaptative_wheel(self):
        pmin = 0.1
        u_selection = list(map(list, zip(*self.stats['utility'])))
        n = len(self.parameters['selection'][2])
        print(u_selection)
        if u_selection:
            sum_uk = sum([u for u, _ in u_selection[0]])
            for i in range(len(self.parameters['selection'][2])):
                m = self.parameters['selection'][2][i][1][0]
                u_n = sum([u for u, me in u_selection[0] if m in me])
                self.parameters['selection'][2][i][0] = pmin + (1 - n * pmin) * (u_n / sum_uk if sum_uk != 0 else 1)
            print(self.parameters['selection'][2])
        self.select_adaptative_fixe()

    def select_adaptative_poursuit(self):
        pass
        # pmin = 0.1
        # n = len(self.parameters['selection'][2])
        # pmax = 1 - (n - 1) * pmin
        # u_selection = list(map(list, zip(*self.stats['utility'])))
        # print(u_selection)
        # pbest = max(u_selection)
        # if u_selection:
        #     sum_uk = sum([u for u, _ in u_selection[0]])
        #     for i in range(len(self.parameters['selection'][2])):
        #         m = self.parameters['selection'][2][i][1][0]
        #         u_n = sum([u for u, me in u_selection[0] if m in me])
        #         self.parameters['selection'][2][i][0] = pmin + (1 - n * pmin) * (u_n / sum_uk if sum_uk != 0 else 1)
        #     # print(self.parameters['selection'][2])
        # self.select_adaptative_fixe()

    def select_adaptative_ucb(self):
        u_selection = list(map(list, zip(*self.stats['utility'])))
        if u_selection:
            past_selected = [m[1] for m in u_selection[0]]
            # sum_uk = sum([u for u, _ in u_selection[0]])
            for i in range(len(self.parameters['selection'][2])):
                m = self.parameters['selection'][2][i][1][0]
                nb_i = past_selected.count(m)
                u_i = sum([u for u, me in u_selection[0] if m in me])
                # print(m)
                # print(u_i)
                # print( math.sqrt(
                #     2 * math.log(len(past_selected)) / (nb_i if nb_i != 0 else 1)))
                self.parameters['selection'][2][i][0] = u_i + math.sqrt(
                    2 * math.log(len(past_selected)) / (nb_i if nb_i != 0 else 1))
            print(self.parameters['selection'][2])
        self.select_adaptative_fixe()

    def select_adaptative_dmab(self):
        pass

    def select_random(self):
        selected = random.sample(self.individuals, self.nb_select)
        self.stats['utility'].append([[[f for _, f, _ in selected], 'select_random']])
        self.selected = copy.deepcopy([i for i, _, _ in selected])

    def select_best(self):
        selected = self.individuals[0:self.nb_select]
        self.stats['utility'].append([[[f for _, f, _ in selected], 'select_best']])
        self.selected = copy.deepcopy([i for i, _, _ in selected])

    def select_tournament(self, parameter=None):
        if self.parameters['selection'][0] == 'select_tournament':
            nb_winners = self.parameters['selection'][1]
            nb_players = self.parameters['selection'][2]
        else:
            nb_winners = parameter[1]
            nb_players = parameter[2]
        self.stats['utility'].append([[[], 'select_tournament']])
        while len(self.selected) < self.nb_select:
            selected = random.sample(self.individuals, nb_players)
            selected.sort(key=lambda i: i[1], reverse=True)
            selected = selected[0:nb_winners]
            self.stats['utility'][-1][0][0].extend([f for _, f, _ in selected])
            self.selected.extend(copy.deepcopy([i for i, _, _ in selected]))
        self.selected = self.selected[0:self.nb_select]

    def select_wheel(self):
        individuals, fits, _ = list(zip(*self.individuals))
        total = sum(fits)
        wheel = [1] * len(fits) if total == 0 else [f / total for f in fits]
        probabilities = [sum(wheel[:i + 1]) for i in range(len(wheel))]
        self.stats['utility'].append([[[], 'select_wheel']])
        for n in range(self.nb_select):
            r = random.random()
            for (i, individual) in enumerate(individuals):
                if r <= probabilities[i]:
                    self.stats['utility'][-1][0][0].append(individual.fitness())
                    self.selected.append(copy.deepcopy(individual))
                    break

    ###############################################################
    #                       Crossover                             #
    ###############################################################

    def crossover(self):
        self.crossed = []
        if self.parameters['proportion crossover'] == 0:
            self.crossed = self.selected
        else:
            switch = {
                'mono-point': self.crossover_monopoint,
                'uniforme': self.crossover_uniforme,
            }
            self.stats['utility'][-1].append([[], self.parameters['crossover']])
            for i in range(0, len(self.selected), 2):
                if random.random() <= self.parameters['proportion crossover']:
                    if len(self.selected) <= i + 1:
                        rand = random.choice(self.selected[0:-1])
                        first_child, second_child = switch[self.parameters['crossover']](self.selected[-1], rand)
                    else:
                        first_child, second_child = switch[self.parameters['crossover']](self.selected[i],
                                                                                         self.selected[i + 1])
                else:
                    first_child = self.individual_class(self.parameters)
                    second_child = self.individual_class(self.parameters)
                    first_child.sequence = copy.deepcopy(self.selected[i][::])
                    # the second child will be a copy of a random parents from the selected parents if the number
                    # of parents is odd
                    sc = random.randrange(len(self.selected[0:-1])) if len(self.selected) <= i + 1 else i + 1
                    second_child.sequence = copy.deepcopy(self.selected[sc].sequence)
                self.stats['utility'][-1][-1][0].extend([first_child.fitness(), second_child.fitness()])
                self.crossed.extend([first_child, second_child])
        self.stats['utility'][-1][-1][0] = statistics.mean(self.stats['utility'][-1][-1][0])

    def crossover_monopoint(self, i1, i2):
        first_child = self.individual_class(self.parameters)
        second_child = self.individual_class(self.parameters)
        rand = random.randint(1, len(i1.sequence))
        first_child[::] = i1[0:rand] + i2[rand:]
        second_child[::] = i2[0:rand] + i1[rand:]
        return first_child, second_child

    def crossover_uniforme(self, i1, i2):
        first_child = self.individual_class(self.parameters)
        second_child = self.individual_class(self.parameters)
        for i in range(self.parameters['chromosome size']):
            first_child[i], second_child[i] = (i1[i], i2[i]) if random.random() <= 0.5 else (i2[i], i1[i])
        return first_child, second_child

    ###############################################################
    #                       Mutation                              #
    ###############################################################

    def mutation(self):
        self.mutated = []
        switch = {
            'n-flip': self.mutation_nfip,
            'bit-flip': self.mutation_bitfip,
            'adaptative': self.mutation_adaptative,
        }
        if self.parameters['mutation'][0] == 'n-flip':
            switch[self.parameters['mutation'][0]](self.parameters['mutation'][1])
        else:
            switch[self.parameters['mutation'][0]]()

    def mutation_adaptative(self):
        switch = {
            'fixed roulette wheel': self.mutation_adaptative_fixe,
            'adaptive roulette wheel': self.mutation_adaptative_wheel,
            'adaptive pursuit': self.mutation_adaptative_poursuit,
            'UCB': self.mutation_adaptative_ucb,
            'DMAB': self.mutation_adaptative_dmab,
        }
        switch[self.parameters['mutation'][1]]()

    def mutation_adaptative_fixe(self):
        proba, select = list(map(list, zip(*self.parameters['mutation'][2])))
        mutation = random.choices(range(len(select)), proba, k=1)[0]
        switch = {
            'n-flip': self.mutation_nfip,
            'bit-flip': self.mutation_bitfip,
        }
        print(self.parameters['mutation'][2][mutation][1])
        if self.parameters['mutation'][2][mutation][1][0] == 'n-flip':
            switch[self.parameters['mutation'][2][mutation][1][0]](self.parameters['mutation'][2][mutation][1][1])
        else:
            switch[self.parameters['mutation'][2][mutation][1][0]]()

    def mutation_adaptative_wheel(self):
        pmin = 0.1
        u_mutation = list(map(list, zip(*self.stats['utility'])))
        n = len(self.parameters['mutation'][2])
        print(u_mutation)
        if u_mutation:
            sum_uk = sum([u for u, _ in u_mutation[2]])
            for i in range(len(self.parameters['mutation'][2])):
                m = self.parameters['mutation'][2][i][1][0]
                u_n = sum([u for u, me in u_mutation[1] if m in me])
                self.parameters['mutation'][2][i][0] = pmin + (1 - n * pmin) * (u_n / sum_uk if sum_uk != 0 else 1)
            print(self.parameters['mutation'][2])
        self.mutation_adaptative_fixe()

    def mutation_adaptative_poursuit(self):
        pass

    def mutation_adaptative_ucb(self):
        u_mutation = list(map(list, zip(*self.stats['utility'])))
        if u_mutation:
            past_selected = [m[1] for m in u_mutation[1]]
            # sum_uk = sum([u for u, _ in u_selection[0]])
            for i in range(len(self.parameters['mutation'][2])):
                m = self.parameters['mutation'][2][i][1][0]
                nb_i = past_selected.count(m)
                u_i = sum([u for u, me in u_mutation[1] if m in me])
                # print(m)
                # print(u_i)
                # print( math.sqrt(
                #     2 * math.log(len(past_selected)) / (nb_i if nb_i != 0 else 1)))
                self.parameters['mutation'][2][i][0] = u_i + math.sqrt(
                    2 * math.log(len(past_selected)) / (nb_i if nb_i != 0 else 1))
            print(self.parameters['mutation'][2])
        self.mutation_adaptative_fixe()

    def mutation_adaptative_dmab(self):
        pass

    def mutation_nfip(self, n):
        for indiv in self.crossed:
            if random.random() <= self.parameters['proportion mutation']:
                for i in random.sample(range(len(indiv.sequence)), n):
                    indiv.sequence[i].bit = 1 - indiv.sequence[i].bit
            self.mutated.append(indiv)

    def mutation_bitfip(self):
        for indiv in self.crossed:
            if random.random() <= self.parameters['proportion mutation']:
                length = self.parameters['chromosome size']
                p = 1 / length
                for i in range(length):
                    if random.random() <= p:
                        indiv.sequence[i].mutate()
            self.mutated.append(indiv)

    ###############################################################
    #                       Insertion                             #
    ###############################################################

    def insertion(self):
        if self.parameters['insertion'] == 'fitness':
            self.sort_individuals_fitness()
        elif self.parameters['insertion'] == 'age':
            self.sort_individuals_age()
        if len(self.mutated):
            self.individuals = self.individuals[:-len(self.mutated)]
            self.stats['utility'][-1].append([[], self.parameters['mutation'][0]])
            for i in self.mutated:
                f = i.fitness()
                self.stats['utility'][-1][-1][0].append(f)
                self.individuals.append((i, f, 0))
        self.stats['utility'][-1][-1][0] = statistics.mean(self.stats['utility'][-1][-1][0])
        selection = self.stats['utility'][-1][0][0]
        crossover = self.stats['utility'][-1][1][0]
        mutation = self.stats['utility'][-1][2][0]
        u_selection = mutation - selection
        u_crossover = crossover - selection
        u_mutation = mutation - crossover
        self.stats['utility'][-1][0][0] = u_selection
        self.stats['utility'][-1][1][0] = u_crossover
        self.stats['utility'][-1][2][0] = u_mutation

    ###############################################################
    #                       Statistic                             #
    ###############################################################

    def statistic(self):
        self.sort_individuals_fitness()
        i = [i for i, _, _ in self.individuals]
        fits = [f for _, f, _ in self.individuals]
        ages = [a for _, _, a in self.individuals]
        self.stats['total_fitness'].append(sum(fits))
        self.stats['max_fitness'].append(fits[0])
        self.stats['min_fitness'].append(fits[-1])
        self.stats['mean_fitness'].append(statistics.mean(fits))
        self.stats['fitness_diversity'].append(len(set(i)))
        self.stats['diversity'].append(len(set(self.individuals)))
        self.stats['max_age'].append(max(ages))
        self.stats['mean_age'].append(statistics.mean(ages))


if __name__ == '__main__':
    param = {
        'configuration name': 'config1',
        'individual': ['algo_gen.individuals.onemax', 'IndividualOneMax'],

        'population size': 100,  # 100 200 500
        'chromosome size': 25,  # 5 10 50 100

        'nb turn max': 10,
        'stop after no change': 10000,  # int(config['nb turn max']*0.10),

        'selection': ['select_tournament', 2, 5],
        # ['adaptative', 'UCB', [[0.25, ['select_random']],
        #                                     [0.25, ['select_best']],
        #                                     [0.25, ['select_tournament', 2, 5]],
        #                                     [0.25, ['select_wheel']]]],
        'proportion selection': 0.04,  # 2 / config['population size']

        'crossover': 'mono-point',  # 'mono-point' 'uniforme'
        'proportion crossover': 1,

        # ['n-flip', 1] ['n-flip', 3] ['n-flip', 5] ['bit-flip']
        # 'mutation': ['n-flip', 3],
        'mutation': ['adaptative',
                     'adaptive roulette wheel',
                     [
                         [0.25, ['n-flip', 1]],
                         [0.25, ['n-flip', 3]],
                         [0.25, ['n-flip', 5]],
                         [0.25, ['bit-flip']]]
                     ],
        'proportion mutation': 0.25,  # 0.1 0.2 0.5 0.8

        'insertion': 'fitness',  # 'age' 'fitness'
    }
    # 'fixed roulette wheel' 'adaptive roulette wheel' 'adaptive pursuit' 'UCB' 'DMAB'

    population = Population(param)
    population.start()

    from algo_gen.tools.plot import show_stats

    show_stats(population.stats)
