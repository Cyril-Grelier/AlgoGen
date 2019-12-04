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
                      f'nb turn max : {self.parameters["nb turn max"]}' \
                      f'\n\tNumber of individuals selected each turns : {self.nb_select}' \
                      f'\n\tSelection : {self.parameters["selection"]}' \
                      f'\n\tCrossover : {self.parameters["crossover"]} (' \
                      f'{self.parameters["proportion crossover"] * 100}%)' \
                      f'\n\tMutation : {self.parameters["mutation"]} (' \
                      f'{self.parameters["proportion mutation"] * 100}%)' \
                      f'\n\tInsertion : {self.parameters["insertion"]}'
        print(description)
        self.method_switch = {
            'selection': {
                'select_random': self.select_random,
                'select_best': self.select_best,
                'select_tournament': self.select_tournament,
                'select_wheel': self.select_wheel,
            },
            'crossover': {
                'mono-point': self.crossover_monopoint,
                'uniforme': self.crossover_uniforme,
            },
            'mutation': {
                '1-flip': self.mutation_1fip,
                '3-flip': self.mutation_3fip,
                '5-flip': self.mutation_5fip,
                'bit-flip': self.mutation_bitfip,
            }
        }

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
    #                       adaptative                            #
    ###############################################################

    def adaptative(self, method):
        switch = {
            'fixed roulette wheel': self.adaptative_fixe,
            'adaptive roulette wheel': self.adaptative_wheel,
            'adaptive pursuit': self.adaptative_poursuit,
            'UCB': self.adaptative_ucb,
        }
        switch[self.parameters[method][1]](method)

    def adaptative_fixe(self, method):
        proba, methode = list(map(list, zip(*self.parameters[method][2])))
        selected_method = random.choices(range(len(methode)), proba, k=1)[0]
        self.method_switch[method][self.parameters[method][2][selected_method][1]]()

    def adaptative_wheel(self, method):
        pmin = 0.1
        switch = {
            'selection': 0,
            'crossover': 1,
            'mutation': 2,
        }
        rank = switch[method]
        utility = list(map(list, zip(*self.stats['utility'][:-1])))
        # [[[0, 'select_tournament']], [[0, 'monopoint']]]
        # self.parameters[method][2] == [[0.25, '1-flip'],[0.25, '3-flip'],[0.25, '5-flip'],[0.25, 'bitflip'],]
        n = len(self.parameters[method][2])
        if len(utility) >= rank:
            sum_uk = sum([u for u, _ in utility[rank]])
            for i in range(len(self.parameters[method][2])):
                m = self.parameters[method][2][i][1]
                print([u for u, me in utility[rank] if m == me])
                u_n = sum([u for u, me in utility[rank] if m == me])
                print(f'{m} : {pmin + (1 - n * pmin) * (u_n / sum_uk if sum_uk != 0 else 0)}')
                self.parameters[method][2][i][0] = pmin + (1 - n * pmin) * (u_n / sum_uk if sum_uk != 0 else 0)
        self.adaptative_fixe(method)

    def adaptative_poursuit(self, method):
        pass
        # pmin = 0.1
        # n = len(self.parameters['selection'][2])
        # pmax = 1 - (n - 1) * pmin
        # u_selection = list(map(list, zip(*self.stats['utility'])))
        # pbest = max(u_selection)
        # if u_selection:
        #     sum_uk = sum([u for u, _ in u_selection[0]])
        #     for i in range(len(self.parameters['selection'][2])):
        #         m = self.parameters['selection'][2][i][1][0]
        #         u_n = sum([u for u, me in u_selection[0] if m in me])
        #         self.parameters['selection'][2][i][0] = pmin + (1 - n * pmin) * (u_n / sum_uk if sum_uk != 0 else 1)
        # self.select_adaptative_fixe()

    def adaptative_ucb(self, method):
        switch = {
            'selection': 0,
            'crossover': 1,
            'mutation': 2,
        }
        rank = switch[method]
        utility = list(map(list, zip(*self.stats['utility'][:-1])))
        if len(utility) >= rank:
            past_selected = utility[rank]
            exploitations = []
            explorations = []
            for i in range(len(self.parameters[method][2])):
                methode = self.parameters[method][2][i][1]
                nb_i = [m[1] for m in past_selected].count(methode)
                print(f'nb_i {methode} : {nb_i}')
                exploitations.append(sum([u for u, me in past_selected if methode in me]))
                explorations.append(math.sqrt(2 * math.log(len(past_selected)) / (nb_i + 1)))  # if nb_i != 0 else 1)))
            # min_exploi = min(exploitations)
            # max_exploi = max(exploitations)
            # min_explor = min(explorations)
            # max_explor = max(explorations)
            # normalize_exploitations = [
            #     ((x - min_exploi) / ((max_exploi - min_exploi) if (max_exploi - min_exploi) != 0 else 1)) for x in
            #     exploitations]
            # normalize_explorations = [
            #     ((x - min_explor) / ((max_explor - min_explor) if (max_explor - min_explor) != 0 else 1)) for x in
            #     explorations]
            print('!' * 80)
            print(f'exploitations {exploitations}')
            # print(f'normalize_exploitations {normalize_exploitations}')
            print(f'explorations {explorations}')
            # print(f'normalize_explorations {normalize_explorations}')
            for i in range(len(self.parameters[method][2])):
                self.parameters[method][2][i][0] = exploitations[i] + explorations[i]
            print(f'self.parameters[method][2] {self.parameters[method][2]}')
        maxi = max(self.parameters[method][2], key=lambda e: e[0])[0]
        indices = [i for i, v in enumerate(self.parameters[method][2]) if v[0] == maxi]
        choosen = random.choice(indices)
        self.method_switch[method][self.parameters[method][2][choosen][1]]()

    def adaptative_dmab(self):
        pass

    ###############################################################
    #                       Selection                             #
    ###############################################################

    def selection(self):
        """
        chose the type of selection to use and select individuals
        :return:
        """
        self.selected = []
        if self.parameters['selection'][0] == 'adaptative':
            self.adaptative('selection')
        else:
            switch = {
                'select_random': self.select_random,
                'select_best': self.select_best,
                'select_tournament': self.select_tournament,
                'select_wheel': self.select_wheel,
            }
            switch[self.parameters['selection'][0]]()
        self.stats['utility'][-1][0][0] = statistics.mean(self.stats['utility'][-1][0][0])

    def select_random(self):
        selected = random.sample(self.individuals, self.nb_select)
        self.stats['utility'].append([[[f for _, f, _ in selected], 'select_random']])
        self.selected = copy.deepcopy([i for i, _, _ in selected])

    def select_best(self):
        selected = self.individuals[0:self.nb_select]
        self.stats['utility'].append([[[f for _, f, _ in selected], 'select_best']])
        self.selected = copy.deepcopy([i for i, _, _ in selected])

    def select_tournament(self):
        nb_winners = 2
        nb_players = 5
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
        if self.parameters['crossover'][0] == 'adaptative':
            self.adaptative('crossover')
        else:
            switch = {
                'mono-point': self.crossover_monopoint,
                'uniforme': self.crossover_uniforme,
            }
            switch[self.parameters['crossover'][0]]()

    def crossover_monopoint(self):
        self.stats['utility'][-1].append([[], 'monopoint'])
        for i in range(0, len(self.selected), 2):
            first_child = self.individual_class(self.parameters)
            second_child = self.individual_class(self.parameters)
            if random.random() <= self.parameters['proportion crossover']:
                if len(self.selected) <= i + 1:
                    rand = random.choice(self.selected[0:-1])
                    i1 = self.selected[-1]
                    i2 = rand
                else:
                    i1 = self.selected[i]
                    i2 = self.selected[i + 1]
                rand = random.randint(1, len(i1.sequence))
                first_child[::] = i1[0:rand] + i2[rand:]
                second_child[::] = i2[0:rand] + i1[rand:]
            else:
                first_child.sequence = copy.deepcopy(self.selected[i][::])
                # the second child will be a copy of a random parents from the selected parents if the number
                # of parents is odd
                sc = random.randrange(len(self.selected[0:-1])) if len(self.selected) <= i + 1 else i + 1
                second_child.sequence = copy.deepcopy(self.selected[sc].sequence)
            self.stats['utility'][-1][-1][0].extend([first_child.fitness(), second_child.fitness()])
            self.crossed.extend([first_child, second_child])
        self.stats['utility'][-1][-1][0] = statistics.mean(self.stats['utility'][-1][-1][0])

    def crossover_uniforme(self):
        self.stats['utility'][-1].append([[], 'uniforme'])
        for i in range(0, len(self.selected), 2):
            first_child = self.individual_class(self.parameters)
            second_child = self.individual_class(self.parameters)
            if random.random() <= self.parameters['proportion crossover']:
                if len(self.selected) <= i + 1:
                    rand = random.choice(self.selected[0:-1])
                    i1 = self.selected[-1]
                    i2 = rand
                else:
                    i1 = self.selected[i]
                    i2 = self.selected[i + 1]
                for j in range(self.parameters['chromosome size']):
                    first_child[j], second_child[j] = (i1[j], i2[j]) if random.random() <= 0.5 else (i2[j], i1[j])
            else:
                first_child.sequence = copy.deepcopy(self.selected[i][::])
                # the second child will be a copy of a random parents from the selected parents if the number
                # of parents is odd
                sc = random.randrange(len(self.selected[0:-1])) if len(self.selected) <= i + 1 else i + 1
                second_child.sequence = copy.deepcopy(self.selected[sc].sequence)
            self.stats['utility'][-1][-1][0].extend([first_child.fitness(), second_child.fitness()])
            self.crossed.extend([first_child, second_child])
        self.stats['utility'][-1][-1][0] = statistics.mean(self.stats['utility'][-1][-1][0])

    ###############################################################
    #                       Mutation                              #
    ###############################################################

    def mutation(self):
        self.mutated = []
        if self.parameters['mutation'][0] == 'adaptative':
            self.adaptative('mutation')
        else:
            switch = {
                '1-flip': self.mutation_1fip,
                '3-flip': self.mutation_3fip,
                '5-flip': self.mutation_5fip,
                'bit-flip': self.mutation_bitfip,
            }
            switch[self.parameters['mutation'][0]]()

    def mutation_1fip(self):
        self.stats['utility'][-1].append([[], '1-flip'])
        for indiv in self.crossed:
            if random.random() <= self.parameters['proportion mutation']:
                for i in random.sample(range(len(indiv.sequence)), 1):
                    indiv.sequence[i].mutate()
            self.mutated.append(indiv)

    def mutation_3fip(self):
        self.stats['utility'][-1].append([[], '3-flip'])
        for indiv in self.crossed:
            if random.random() <= self.parameters['proportion mutation']:
                for i in random.sample(range(len(indiv.sequence)), 3):
                    indiv.sequence[i].mutate()
            self.mutated.append(indiv)

    def mutation_5fip(self):
        self.stats['utility'][-1].append([[], '5-flip'])
        for indiv in self.crossed:
            if random.random() <= self.parameters['proportion mutation']:
                for i in random.sample(range(len(indiv.sequence)), 5):
                    indiv.sequence[i].mutate()
            self.mutated.append(indiv)

    def mutation_bitfip(self):
        self.stats['utility'][-1].append([[], 'bit-flip'])
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
            for i in self.mutated:
                f = i.fitness()
                self.stats['utility'][-1][-1][0].append(f)
                self.individuals.append((i, f, 0))
        self.stats['utility'][-1][-1][0] = statistics.mean(self.stats['utility'][-1][-1][0])
        selection = self.stats['utility'][-1][0][0]
        crossover = self.stats['utility'][-1][1][0]
        mutation = self.stats['utility'][-1][2][0]
        u_selection = (mutation - selection) / self.parameters['chromosome size']
        u_crossover = (crossover - selection) / self.parameters['chromosome size']
        u_mutation = (mutation - crossover) / self.parameters['chromosome size']
        self.stats['utility'][-1][0][0] = u_selection
        self.stats['utility'][-1][1][0] = u_crossover
        self.stats['utility'][-1][2][0] = u_mutation
        # print(f'self.parameters["mutation"][2] {self.parameters["mutation"][2]}')
        # print(f'self.stats["utility"][-1][2][1] {self.stats["utility"][-1][2][1]}')
        # print(f'self.stats["utility"][-1][2][0] {self.stats["utility"][-1][2][0]}')

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

        'population size': 50,  # 100 200 500
        'chromosome size': 100,  # 5 10 50 100

        'nb turn max': 1000,
        'stop after no change': 50,  # int(config['nb turn max']*0.10),

        'selection':
            ['select_tournament'],
        #     ['adaptative',
        #      'UCB',
        #      [
        #          [0.25, 'select_random'],
        #          [0.25, 'select_best'],
        #          [0.25, 'select_tournament'],
        #          [0.25, 'select_wheel']
        #      ]],
        'proportion selection': 0.04,  # 2 / population_size

        'crossover':
        ['mono-point'],
        #     ['adaptative',
        #      'UCB',
        #      [
        #          [0.25, 'mono-point'],
        #          [0.25, 'uniforme'],
        #      ]],
        'proportion crossover': 0,

        'mutation':
        # ['3-flip'],
            ['adaptative',
             'UCB',
             # 'fixed roulette wheel' 'adaptive roulette wheel' 'adaptive pursuit' 'UCB'
             [
                 [0.25, '1-flip'],
                 [0.25, '3-flip'],
                 [0.25, '5-flip'],
                 [0.25, 'bit-flip']
             ], ],
        'proportion mutation': 1,  # 0.1 0.2 0.5 0.8

        'insertion': 'fitness',  # 'age' 'fitness'
    }
    # 'fixed roulette wheel' 'adaptive roulette wheel' 'adaptive pursuit' 'UCB' 'DMAB'

    population = Population(param)
    population.start()

    from algo_gen.tools.plot import show_stats

    show_stats(population.stats)
