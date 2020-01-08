import math
import random
import statistics
from abc import ABC, abstractmethod
from copy import deepcopy


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

    def __init__(self, parameters, empty=False):
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
    def crossover(self, other):
        """
        crossover used if you use the crossover 'individual
        :param other: the second parent for the crossover
        :type other: Individual
        :return: the
        """
        pass

    @abstractmethod
    def mutate(self):
        """
        mutation called if you use the mutation 'individual'
        if you need to mutate on the individual instead of the gene
        :return: None
        """
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


class IndividualPermutation(Individual, ABC):
    @abstractmethod
    def basic_order(self):
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

        if 'function_each_turn' in self.parameters:
            self.each_turn_fct = self.parameters['function_each_turn']
        else:
            self.each_turn_fct = None

        if 'function_end' in self.parameters:
            self.end_fct = self.parameters['function_end']
        else:
            self.end_fct = None

        if 'termination_condition' in self.parameters:
            self.final_condition = self.parameters['termination_condition']
        else:
            self.final_condition = lambda pop: not (pop.nb_turns == pop.parameters['nb turn max'])

        self.individual_class = parameters['individual']

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
                      f'\n\tnb turn max : {self.parameters["nb turn max"]}' \
                      f'\n\tNumber of individuals selected each turns : {self.nb_select}' \
                      f'\n\tSelection : {self.parameters["selection"]}' \
                      f'\n\tCrossover : {self.parameters["crossover"]} (' \
                      f'{self.parameters["proportion crossover"] * 100}%)' \
                      f'\n\tMutation : {self.parameters["mutation"]} (' \
                      f'{self.parameters["proportion mutation"] * 100}%)' \
                      f'\n\tInsertion : {self.parameters["insertion"]}'
        print(description)
        self.method_switch = dict()
        self.method_switch['selection'] = {
            'select_random': self.select_random,
            'select_best': self.select_best,
            'select_tournament': self.select_tournament,
            'select_wheel': self.select_wheel,
        }
        if isinstance(self.individuals[0][0], IndividualPermutation):
            self.method_switch['crossover'] = {
                'individual': self.crossover_individual,
                'order 1': self.crossover_order_1,
                'pmx': self.crossover_pmx,
            }
            self.method_switch['mutation'] = {
                'individual': self.mutation_individu,
                'insert': self.mutation_insert,
                'swap': self.mutation_swap,
                'inversion': self.mutation_inversion,
                'scramble': self.mutation_scramble,
            }
        else:
            self.method_switch['crossover'] = {
                'individual': self.crossover_individual,
                'mono-point': self.crossover_monopoint,
                'multipoint': self.crossover_multipoint,
                'uniforme': self.crossover_uniforme,
            }
            self.method_switch['mutation'] = {
                'individual': self.mutation_individu,
                '1-flip': self.mutation_1fip,
                '3-flip': self.mutation_3fip,
                '5-flip': self.mutation_5fip,
                'bit-flip': self.mutation_bitfip,
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

    def start(self):
        self.statistic()
        while self.final_condition(self):
            if self.each_turn_fct:
                self.each_turn_fct(self)
            self.population_get_older()
            self.sort_individuals_fitness()
            self.turn()
            self.statistic()
        if self.end_fct:
            self.end_fct(self)

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
            'adaptive roulette wheel': self.adaptive_roulette_wheel,
            'adaptive pursuit': self.adaptative_poursuit,
            'UCB': self.adaptative_ucb,
        }
        switch[self.parameters[method][1]](method)

    def adaptative_fixe(self, method):
        proba, methode = list(map(list, zip(*self.parameters[method][2])))
        selected_method = random.choices(range(len(methode)), proba, k=1)[0]
        self.method_switch[method][self.parameters[method][2][selected_method][1]]()

    def adaptive_roulette_wheel(self, method):
        pmin = self.parameters[method][3]
        switch = {
            'selection': 0,
            'crossover': 1,
            'mutation': 2,
        }
        rank = switch[method]
        utility = list(map(list, zip(*self.stats['utility'][:-1])))
        utility = utility[-5:]
        n = len(self.parameters[method][2])
        if len(utility) >= rank:
            sum_uk = sum([u for u, _ in utility[rank]])
            for i in range(len(self.parameters[method][2])):
                m = self.parameters[method][2][i][1]
                u_n = sum([u for u, me in utility[rank] if m == me])
                self.parameters[method][2][i][0] = pmin + (1 - n * pmin) * (u_n / sum_uk if sum_uk != 0 else 0)
        self.adaptative_fixe(method)

    def adaptative_poursuit(self, method):
        switch = {
            'selection': 0,
            'crossover': 1,
            'mutation': 2,
        }
        rank = switch[method]
        utility = list(map(list, zip(*self.stats['utility'][:-1])))
        # if len(utility) > 50:
        #     utility = utility[-100:]
        #     if len(utility) % 50 == 0:
        #         for i in range(len(self.parameters[method][2])):
        #             self.parameters[method][2][i][0] = 1 / len(self.parameters[method][2])
        if len(utility) >= rank:
            pmin = self.parameters[method][3]
            pmax = 1 - (len(self.parameters[method][2]) - 1) * pmin
            beta = self.parameters[method][4]
            past_selected = utility[rank]
            utilities = []
            for i in range(len(self.parameters[method][2])):
                methode = self.parameters[method][2][i][1]
                nb_i = [m[1] for m in past_selected].count(methode)
                utilities.append(sum([u for u, me in past_selected if methode in me]))
            maxi = max(self.parameters[method][2], key=lambda e: e[0])[0]
            indices = [i for i, v in enumerate(self.parameters[method][2]) if v[0] == maxi]
            best = random.choice(indices)
            for i in range(len(self.parameters[method][2])):
                tminus1 = self.parameters[method][2][i][0]
                if i == best:
                    self.parameters[method][2][i][0] = tminus1 + beta * (pmax - tminus1)
                else:
                    self.parameters[method][2][i][0] = tminus1 + beta * (pmin - tminus1)
        self.adaptative_fixe(method)

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
                exploitations.append(sum([u for u, me in past_selected if methode in me]))
                explorations.append(math.sqrt(2 * math.log(len(past_selected)) / (nb_i + 1)))
            for i in range(len(self.parameters[method][2])):
                self.parameters[method][2][i][0] = exploitations[i] + explorations[i]
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
        self.selected = deepcopy([i for i, _, _ in selected])

    def select_best(self):
        selected = self.individuals[0:self.nb_select]
        self.stats['utility'].append([[[f for _, f, _ in selected], 'select_best']])
        self.selected = deepcopy([i for i, _, _ in selected])

    def select_tournament(self):
        nb_winners = 2
        nb_players = 5
        self.stats['utility'].append([[[], 'select_tournament']])
        while len(self.selected) < self.nb_select:
            selected = random.sample(self.individuals, nb_players)
            selected.sort(key=lambda i: i[1], reverse=True)
            selected = selected[0:nb_winners]
            self.stats['utility'][-1][0][0].extend([f for _, f, _ in selected])
            self.selected.extend(deepcopy([i for i, _, _ in selected]))
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
                    self.selected.append(deepcopy(individual))
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
                'individual': self.crossover_individual,
                'mono-point': self.crossover_monopoint,
                'multipoint': self.crossover_multipoint,
                'uniforme': self.crossover_uniforme,
                'order 1': self.crossover_order_1,
                'pmx': self.crossover_pmx,
            }
            switch[self.parameters['crossover'][0]]()

    def crossover_individual(self):
        self.stats['utility'][-1].append([[], 'individual'])
        for i in range(0, len(self.selected), 2):
            first_child, second_child = None, None
            if random.random() <= self.parameters['proportion crossover']:
                if len(self.selected) <= i + 1:
                    rand = random.choice(self.selected[0:-1])
                    i1 = self.selected[-1]
                    i2 = rand
                else:
                    i1 = self.selected[i]
                    i2 = self.selected[i + 1]
                first_child, second_child = i1.crossover(i2)
            else:
                first_child.sequence = deepcopy(self.selected[i][::])
                # the second child will be a copy of a random parents from the selected parents if the number
                # of parents is odd
                sc = random.randrange(len(self.selected[0:-1])) if len(self.selected) <= i + 1 else i + 1
                second_child.sequence = deepcopy(self.selected[sc].sequence)
            self.stats['utility'][-1][-1][0].extend([first_child.fitness(), second_child.fitness()])
            self.crossed.extend([first_child, second_child])
        self.stats['utility'][-1][-1][0] = statistics.mean(self.stats['utility'][-1][-1][0])

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
                first_child.sequence = deepcopy(self.selected[i][::])
                # the second child will be a copy of a random parents from the selected parents if the number
                # of parents is odd
                sc = random.randrange(len(self.selected[0:-1])) if len(self.selected) <= i + 1 else i + 1
                second_child.sequence = deepcopy(self.selected[sc].sequence)
            self.stats['utility'][-1][-1][0].extend([first_child.fitness(), second_child.fitness()])
            self.crossed.extend([first_child, second_child])
        self.stats['utility'][-1][-1][0] = statistics.mean(self.stats['utility'][-1][-1][0])

    def crossover_multipoint(self):
        self.stats['utility'][-1].append([[], 'multipoint'])
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
                rand1 = random.randint(1, len(i1.sequence) - 2)
                rand2 = random.randint(rand1, len(i1.sequence))
                first_child[::] = deepcopy(i1[0:rand1]) + deepcopy(i2[rand1:rand2]) + deepcopy(i1[rand2:])
                second_child[::] = deepcopy(i2[0:rand1]) + deepcopy(i1[rand1:rand2]) + deepcopy(i2[rand2:])
            else:
                first_child.sequence = deepcopy(self.selected[i][::])
                # the second child will be a copy of a random parents from the selected parents if the number
                # of parents is odd
                sc = random.randrange(len(self.selected[0:-1])) if len(self.selected) <= i + 1 else i + 1
                second_child.sequence = deepcopy(self.selected[sc].sequence)
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
                first_child.sequence = deepcopy(self.selected[i][::])
                # the second child will be a copy of a random parents from the selected parents if the number
                # of parents is odd
                sc = random.randrange(len(self.selected[0:-1])) if len(self.selected) <= i + 1 else i + 1
                second_child.sequence = deepcopy(self.selected[sc].sequence)
            self.stats['utility'][-1][-1][0].extend([first_child.fitness(), second_child.fitness()])
            self.crossed.extend([first_child, second_child])
        self.stats['utility'][-1][-1][0] = statistics.mean(self.stats['utility'][-1][-1][0])

    def crossover_order_1(self):
        """

        """
        # prendre deux val random, la 2nd plus grande que la 1ere
        # copie du parent 1 dans le child 1
        self.stats['utility'][-1].append([[], 'order 1'])
        for i in range(0, len(self.selected), 2):
            first_child = self.individual_class(self.parameters, empty=True)
            second_child = self.individual_class(self.parameters, empty=True)
            if random.random() <= self.parameters['proportion crossover']:
                if len(self.selected) <= i + 1:
                    rand = random.choice(self.selected[0:-1])
                    i1 = self.selected[-1]
                    i2 = rand
                else:
                    i1 = self.selected[i]
                    i2 = self.selected[i + 1]
                size = len(first_child.sequence)
                rand1 = random.randint(0, size - 2)
                rand2 = random.randint(rand1, size)
                first_child[rand1:rand2] = deepcopy(i1[rand1:rand2])
                second_child[rand1:rand2] = deepcopy(i2[rand1:rand2])
                k = rand2
                j = rand2
                i = 0
                while None in first_child.sequence:
                    i += 1
                    k %= size
                    j %= size
                    if not i2[k] in first_child.sequence:
                        first_child[j] = deepcopy(i2[k])
                        j += 1
                    k += 1
                i = 0
                k = rand2
                j = rand2
                while None in second_child.sequence:
                    i += 1
                    k %= size
                    j %= size
                    if not i1[k] in second_child.sequence:
                        second_child[j] = deepcopy(i1[k])
                        j += 1
                    k += 1
            else:
                first_child.sequence = deepcopy(self.selected[i][::])
                # the second child will be a copy of a random parents from the selected parents if the number
                # of parents is odd
                sc = random.randrange(len(self.selected[0:-1])) if len(self.selected) <= i + 1 else i + 1
                second_child.sequence = deepcopy(self.selected[sc].sequence)
            self.stats['utility'][-1][-1][0].extend([first_child.fitness(), second_child.fitness()])
            self.crossed.extend([first_child, second_child])
        self.stats['utility'][-1][-1][0] = statistics.mean(self.stats['utility'][-1][-1][0])

    def crossover_pmx(self):
        self.stats['utility'][-1].append([[], 'pmx'])
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

                size = len(first_child.sequence)
                rand1 = random.randint(0, size - 2)
                rand2 = random.randint(rand1 + 1, size - 1)
                indexes = i1.basic_order()
                l1 = [indexes.index(i1[j]) for j in range(size)]
                l2 = [indexes.index(i2[j]) for j in range(size)]

                p1 = [None] * size
                p2 = [None] * size
                for j in range(size):
                    p1[l1[j]] = j
                    p2[l2[j]] = j
                for j in range(rand1, rand2):
                    val1 = l1[j]
                    val2 = l2[j]
                    l1[j], l1[p1[val2]] = val2, val1
                    p1[val1], p1[val2] = p1[val2], p1[val1]
                    l2[j], l2[p2[val1]] = val1, val2
                    p2[val1], p2[val2] = p2[val2], p2[val1]

                c1 = [indexes[l1[j]] for j in range(size)]
                c2 = [indexes[l2[j]] for j in range(size)]
                first_child.sequence = deepcopy(c1)
                second_child.sequence = deepcopy(c2)
            else:
                first_child.sequence = deepcopy(self.selected[i][::])
                # the second child will be a copy of a random parents from the selected parents if the number
                # of parents is odd
                sc = random.randrange(len(self.selected[0:-1])) if len(self.selected) <= i + 1 else i + 1
                second_child.sequence = deepcopy(self.selected[sc].sequence)
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
                'individual': self.mutation_individu,
                '1-flip': self.mutation_1fip,
                '3-flip': self.mutation_3fip,
                '5-flip': self.mutation_5fip,
                'bit-flip': self.mutation_bitfip,
                'insert': self.mutation_insert,
                'swap': self.mutation_swap,
                'inversion': self.mutation_inversion,
                'scramble': self.mutation_scramble,
            }
            switch[self.parameters['mutation'][0]]()

    def mutation_individu(self):
        self.stats['utility'][-1].append([[], 'individual'])
        for indiv in self.crossed:
            indiv.mutate()
            self.mutated.append(indiv)

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

    def mutation_insert(self):
        """
        prendre deux points, le second devient le suivant du premier
        """
        self.stats['utility'][-1].append([[], 'insert'])
        for indiv in self.crossed:
            if random.random() <= self.parameters['proportion mutation']:
                length = self.parameters['chromosome size']
                rand1 = random.randint(0, length - 3)
                rand2 = random.randint(rand1 + 1, length - 1)
                indiv[rand1 + 1], indiv[rand2] = indiv[rand2], indiv[rand1 + 1]
            self.mutated.append(indiv)

    def mutation_swap(self):
        """
        prendre deux points et les echanger
        """
        self.stats['utility'][-1].append([[], 'swap'])
        for indiv in self.crossed:
            if random.random() <= self.parameters['proportion mutation']:
                length = self.parameters['chromosome size']
                rand1 = random.randint(0, length - 3)
                rand2 = random.randint(rand1 + 1, length - 1)
                indiv[rand1], indiv[rand2] = indiv[rand2], indiv[rand1]
            self.mutated.append(indiv)

    def mutation_inversion(self):
        """
        inverser toutes les valeurs entre deux points
        """
        self.stats['utility'][-1].append([[], 'invertion'])
        for indiv in self.crossed:
            if random.random() <= self.parameters['proportion mutation']:
                length = self.parameters['chromosome size']
                rand1 = random.randint(0, length - 3)
                rand2 = random.randint(rand1 + 1, length - 1)
                indiv[rand1:rand2] = indiv[rand1:rand2][::-1]
            self.mutated.append(indiv)

    def mutation_scramble(self):
        """
        melanger toutes les valeurs entre deux points
        """
        self.stats['utility'][-1].append([[], 'scramble'])
        for indiv in self.crossed:
            if random.random() <= self.parameters['proportion mutation']:
                length = self.parameters['chromosome size']
                rand1 = random.randint(0, length - 3)
                rand2 = random.randint(rand1 + 1, length - 1)
                randlist = indiv[rand1:rand2]
                random.shuffle(randlist)
                indiv[rand1:rand2] = randlist
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


def mainpermut():
    number_of_team = 6
    from algo_gen.individuals.STS import IndividualSTS
    parameters = {
        'configuration name': 'config1',
        'individual': IndividualSTS,
        'number of team': number_of_team,
        'population size': 100,
        'chromosome size': (number_of_team - 1) * (number_of_team // 2),

        'nb turn max': 10000,
        'stop after no change': 500,

        'selection':
            ['select_tournament'],
        'proportion selection': 0.1,  # 0.04,  # 2 / population_size

        'crossover':
            ['pmx'],
        'proportion crossover': 1,

        'mutation':
            ['individual'],

        # 'insert'
        # 'swap'
        # 'inversion'
        # 'scramble'

        'proportion mutation': 0.2,  # 0.1 0.2 0.5 0.8

        'insertion': 'age',  # 'age' 'fitness'
    }
    population = Population(parameters)
    population.start()
    from algo_gen.tools.plot import show_stats
    show_stats(population.stats)


def main():
    from algo_gen.individuals.onemax import IndividualOneMax

    def final_condition(population):
        if population.nb_turns >= population.parameters['stop after no change']:
            last_max = population.stats['max_fitness'][-population.parameters['stop after no change']:]
            # last_min = self.stats['min_fitness'][-self.parameters['stop after no change']:]
            max_change = not all(x >= y for x, y in zip(last_max, last_max[1:]))
            # min_change = not all(x >= y for x, y in zip(last_min, last_min[1:]))
            return (not (population.nb_turns == population.parameters['nb turn max'])) and max_change
            # and not (not max_change and not min_change)
        else:
            return not (population.nb_turns == population.parameters['nb turn max'])

    def function_each_turn(population):
        pass

    def function_end(population):
        print(f'fitness max : {population.stats["max_fitness"][-1]}')

    param = {
        'configuration name': 'config1',
        'individual': IndividualOneMax,

        'population size': 50,  # 100 200 500
        'chromosome size': 100,  # 5 10 50 100

        'termination_condition': final_condition,

        'function_each_turn': function_each_turn,
        'function_end': function_end,

        'nb turn max': 5000,
        'stop after no change': 5000000,  # int(config['nb turn max']*0.10),

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
        'proportion selection': 0.04,  # 0.04,  # 2 / population_size

        'crossover':
            ['individual'],
        #     ['adaptative',
        #      'UCB',
        #      [
        #          [0.25, 'mono-point'],
        #          [0.25, 'uniforme'],
        #      ],
        #      0.5],
        'proportion crossover': 1,

        'mutation':
            ['individual'],
        #     ['adaptative',
        #      'UCB',
        #      # 'fixed roulette wheel' 'adaptive roulette wheel' 'adaptive pursuit' 'UCB'
        #      [
        #          [0.25, '1-flip'],
        #          [0.25, '3-flip'],
        #          [0.25, '5-flip'],
        #          [0.25, 'bit-flip']
        #      ],
        #      0.05,  # pmin for adaptive roulette wheel and adaptive poursuite
        #      0.5,  # beta for adaptive poursuit
        #      ],
        'proportion mutation': 1,  # 0.1 0.2 0.5 0.8

        'insertion': 'fitness',  # 'age' 'fitness'
    }
    # 'fixed roulette wheel' 'adaptive roulette wheel' 'adaptive pursuit' 'UCB' 'DMAB'

    diff_mutation = [
        ['adaptative', 'fixed roulette wheel',
         [[0.25, '1-flip'], [0.25, '3-flip'], [0.25, '5-flip'], [0.25, 'bit-flip']], ],
        ['adaptative', 'adaptive roulette wheel',
         [[0.25, '1-flip'], [0.25, '3-flip'], [0.25, '5-flip'], [0.25, 'bit-flip']], 0.05, ],
        ['adaptative', 'adaptive pursuit',
         [[0.25, '1-flip'], [0.25, '3-flip'], [0.25, '5-flip'], [0.25, 'bit-flip']], 0.05, 0.8, ],
        ['adaptative', 'UCB',
         [[0.25, '1-flip'], [0.25, '3-flip'], [0.25, '5-flip'], [0.25, 'bit-flip']], ],
    ]

    for mu in diff_mutation:
        param['mutation'] = mu
        population = Population(param)
        population.start()
        #
        # utility = list(map(list, zip(*population.stats['utility'])))
        # val = list(map(list, zip(*utility[2])))[1]
        #
        # count = {'1-flip': [0],
        #          '3-flip': [0],
        #          '5-flip': [0],
        #          'bit-flip': [0],
        #          }
        # for v in val:
        #     for m in ['1-flip', '3-flip', '5-flip', 'bit-flip']:
        #         count[m].append(count[m][-1] + (1 if v == m else 0))
        #
        # import matplotlib.pyplot as plt
        #
        # fig, ax = plt.subplots(figsize=(10, 10))
        #
        # for m in count.keys():
        #     ax.plot(list(range(len(count[m]))), count[m], label=m)
        # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        # import textwrap
        #
        # plt.title("\n".join(textwrap.wrap(str(population.stats['parameters']['mutation'][1]), 120)))
        # plt.show()
        #
        from algo_gen.tools.plot import show_stats

        show_stats(population.stats)


if __name__ == '__main__':
    main()
    # mainpermut()
