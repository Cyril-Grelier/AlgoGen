import random
from pprint import pformat

from algo_gen.classes import IndividualPermutation


class IndividualSTS(IndividualPermutation):

    def __init__(self, parameters, empty=False):
        super().__init__(parameters, empty=empty)
        if parameters['number of team'] % 2 != 0:
            print("Error, the number of team must be a pair number")
            exit(1)
        self.n = parameters['number of team']
        self.s = self.n - 1
        self.p = self.n // 2
        if empty:
            self.sequence = [None] * (self.s * self.p)
        else:
            self.sequence = [(i, j) for i in range(0, self.n) for j in range(i, self.n) if i != j]
            random.shuffle(self.sequence)

    def fitness(self):
        m = r_to_m(self.sequence, self.s, self.p)
        nb = 0
        for semaine in m:
            present = [-1] * self.n
            for a, b in semaine:
                present[a] += 1
                present[b] += 1
            nb += sum(list(map(abs, present)))  # [abs(ab) for ab in present])
        m_t = transpose(m)
        for periode in m_t:
            present = [0] * self.n
            for a, b in periode:
                present[a] += 1
                present[b] += 1
            nb += sum([0 if i <= 2 else i - 2 for i in present])
        return -nb

    def crossover(self, other):
        return IndividualSTS(self.parameters), IndividualSTS(self.parameters)

    def mutate(self):
        pass

    def basic_order(self):
        return [(i, j) for i in range(0, self.parameters['number of team']) for j in
                range(i, self.parameters['number of team']) if i != j]

    def __eq__(self, other):
        if type(other) != type(self):
            return False
        for a, b in zip(self.sequence, other.sequence):
            if a != b:
                return False
        return True

    def __repr__(self):
        r = ""
        s = self.parameters['number of team'] - 1
        p = self.parameters['number of team'] // 2
        m = [self.sequence[i * p:i * p + p] for i in range(s)]
        r += pformat(m, indent=1)
        r += f" {self.fitness()}"
        return r

    def __hash__(self):
        r = ""
        for g in self.sequence:
            r += str(g[0]) + str(g[1])
        return int(r, 10)


def r_to_m(r, s, p):
    return [r[i * p:i * p + p] for i in range(s)]


def m_to_r(m):
    return [ab for s in m for ab in s]


def transpose(m):
    return list(map(list, zip(*m)))


def permutation(r, i1, i2):
    r[i1], r[i2] = r[i2], r[i1]


def permutation_m(m, i1, i2):
    m[i1[0]][i1[1]], m[i2[0]][i2[1]] = m[i2[0]][i2[1]], m[i1[0]][i1[1]]


def semaines_ok(m, n):
    for semaine in m:
        present = [False] * n
        for a, b in semaine:
            if present[a] or present[b]:
                return False
            present[a] = True
            present[b] = True
    return True


def nombre_incoherences_semaine(m, n):
    nb = 0
    for semaine in m:
        present = [-1] * n
        for a, b in semaine:
            present[a] += 1
            present[b] += 1
        nb += sum(list(map(abs, present)))  # [abs(ab) for ab in present])
    return nb


def matrice_incoherences_semaine(m, n):
    matrice_incoherences = [[0] * (n // 2) for _ in range(n - 1)]
    for num_s, semaine in enumerate(m):
        present = [-1] * n
        for a, b in semaine:
            present[a] += 1
            present[b] += 1
        inco = list(map(abs, present))
        for num_p, ab in enumerate(semaine):
            a, b = ab
            matrice_incoherences[num_s][num_p] += (inco[a] > 0) + (inco[b] > 0)
    return matrice_incoherences


def matrice_incoherences_semaine(m, n):
    matrice_incoherences = [[0] * (n // 2) for _ in range(n - 1)]
    for num_s, semaine in enumerate(m):
        present = [-1] * n
        for a, b in semaine:
            present[a] += 1
            present[b] += 1
        inco = list(map(abs, present))
        for num_p, ab in enumerate(semaine):
            a, b = ab
            matrice_incoherences[num_s][num_p] += (inco[a] > 0) + (inco[b] > 0)
    return matrice_incoherences


def periodes_ok(m_t, n):
    for periode in m_t:
        present = [0] * n
        for a, b in periode:
            present[a] += 1
            present[b] += 1
            if present[a] > 2 or present[b] > 2:
                return False
    return True


def nombre_incoherences_periodes(m_t, n):
    nb = 0
    for periode in m_t:
        present = [0] * n
        for a, b in periode:
            present[a] += 1
            present[b] += 1
        nb += sum([0 if i <= 2 else i - 2 for i in present])
    return nb


def matrice_incoherences_periode(m_t, n):
    matrice_incoherences = [[0] * (n - 1) for _ in range(n // 2)]
    # la matrice est transposee compare avec celle de semaines
    for num_p, periode in enumerate(m_t):
        present = [0] * n
        for a, b in periode:
            present[a] += 1
            present[b] += 1
        inco = [0 if v <= 2 else v - 2 for v in present]
        for num_s, ab in enumerate(periode):
            a, b = ab
            matrice_incoherences[num_p][num_s] += (inco[a] > 0) + (inco[b] > 0)
    return matrice_incoherences


def nombre_incoherences_totales(m, n):
    nb = 0
    for semaine in m:
        present = [-1] * n
        for a, b in semaine:
            present[a] += 1
            present[b] += 1
        nb += sum(list(map(abs, present)))  # [abs(ab) for ab in present])
    m_t = transpose(m)
    for periode in m_t:
        present = [0] * n
        for a, b in periode:
            present[a] += 1
            present[b] += 1
        nb += sum([0 if i <= 2 else i - 2 for i in present])
    return nb


def matrice_incoherences_totales(m, n):
    matrice_incoherences = [[0] * (n // 2) for _ in range(n - 1)]
    for num_s, semaine in enumerate(m):
        present = [-1] * n
        for a, b in semaine:
            present[a] += 1
            present[b] += 1
        inco = list(map(abs, present))
        for num_p, ab in enumerate(semaine):
            a, b = ab
            matrice_incoherences[num_s][num_p] += (inco[a] > 0) + (inco[b] > 0)
    m_t = transpose(m)
    matrice_incoherences = transpose(matrice_incoherences)
    for num_p, periode in enumerate(m_t):
        present = [0] * n
        for a, b in periode:
            present[a] += 1
            present[b] += 1
        inco = [0 if v <= 2 else v - 2 for v in present]
        for num_s, ab in enumerate(periode):
            a, b = ab
            matrice_incoherences[num_p][num_s] += (inco[a] > 0) + (inco[b] > 0)
    return transpose(matrice_incoherences)
