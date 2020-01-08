from random import randint

from algo_gen.classes import Individual, Gene


class GeneOneMax(Gene):

    def __init__(self):
        super().__init__()
        self.bit = 0

    def mutate(self):
        self.bit = 1 - self.bit

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return str(self.bit)


class IndividualOneMax(Individual):

    def __init__(self, parameters, empty=False):
        super().__init__(parameters, empty=empty)
        for _ in range(parameters['chromosome size']):
            self.sequence.append(GeneOneMax())

    def fitness(self):
        return sum(v.bit for v in self.sequence)

    def crossover(self, other):
        fc = IndividualOneMax(self.parameters)
        sc = IndividualOneMax(self.parameters)
        rand = randint(1, len(self.sequence))
        fc[::] = self[0:rand] + other[rand:]
        sc[::] = other[0:rand] + self[rand:]
        return fc, sc

    def mutate(self):
        pass

    def __eq__(self, other):
        if type(other) != type(self):
            return False
        for a, b in zip(self.sequence, other.sequence):
            if a.bit != b.bit:
                return False
        return True

    def __repr__(self):
        r = ""
        for g in self.sequence:
            r += str(g.bit)
        r += f" {self.fitness()}"
        return r

    def __hash__(self):
        r = ""
        for g in self.sequence:
            r += str(g.bit)
        return int(r, 2)
