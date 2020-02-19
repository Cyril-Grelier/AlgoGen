from algo_gen.tools.plot import load_stats

changements = [
    "PopulationSize", "ChromosomeSize", "Selection", "ProportionSelection",
    'Crossover', 'ProportionCrossover', 'Mutation', 'ProportionMutation',
    'Insertion', 'MutationAdaptative', 'CrossoverOrdre1', 'CrossoverPMX', 'Crossover6Ordre1', 'Crossover6PMX'
]
diff_changements = [[5, 25, 50, 75, 100, 200], [5, 25, 50, 75, 100],
                    [
                        'select_random', 'select_best', 'select_tournament',
                        'select_wheel'
                    ], [0.02, 0.1, 0.2, 0.4, 0.6, 0.8, 1],
                    ['mono-point', 'multipoint', 'uniforme'],
                    [0, 0.2, 0.4, 0.6, 0.8, 1],
                    ['1-flip', '3-flip', '5-flip', 'bit-flip'],
                    [0.1, 0.2, 0.4, 0.6, 0.8, 1], ['age', 'fitness'],
                    ['1-flip', 'fixed roulette wheel', 'adaptive roulette wheel', 'adaptive pursuit', 'UCB'],
                    ['insert', 'swap', 'inversion', 'scramble'],
                    ['insert', 'swap', 'inversion', 'scramble'],
                    ['insert', 'swap', 'inversion', 'scramble'],
                    ['insert', 'swap', 'inversion', 'scramble']]

titres = ['Evolution de la recherche en variant la taille de la population',
          'Evolution de la recherche en variant la taille des individu',
          'Evolution de la recherche en variant les operateurs de selection',
          'Evolution de la recherche en variant la proportion de selection',
          'Evolution de la recherche en variant les operateurs de croisement',
          'Evolution de la recherche en variant la proportion de croisement',
          'Evolution de la recherche en variant les operateurs de mutation',
          'Evolution de la recherche en variant les proportions de mutation',
          "Evolution de la recherche en variant l'insertion",
          'Evolution de la recherche en utilisant une mutation adaptative',
          "Evolution de la recherche par permutation 8 equipes avec un croisement d'ordre 1",
          "Evolution de la recherche par permutation 8 equipes avec un croisement PMX",
          "Evolution de la recherche par permutation 6 equipes avec un croisement d'ordre 1",
          "Evolution de la recherche par permutation 6 equipes avec un croisement PMX",
          ]

# for i, cd in enumerate(zip(changements, diff_changements)):
#     c, d = cd
#     stats, label = load_stats(c)
#
#     # plot_all_stats(stats, label, 'Moyennes des recherches de ' + c, d)
#     plot_mean_stats(stats, label, titres[i], d)

stats, label = load_stats('MutationAdaptative')
list_diff = ['1-flip', '3-flip', '5-flip', 'bit-flip']

print(len(stats))
print(label)

for i,s in enumerate(stats):
    if s['parameters']['mutation'] != ['1-flip']:
        if "wheel" in s['parameters']['mutation'][1]:
            print(s['parameters'])
            utility = list(map(list, zip(*s['utility'])))
            val = list(map(list, zip(*utility[2])))[1]
            count = {}
            for d in list_diff:
                count[d] = [0]

            for v in val:
                for m in list_diff:
                    count[m].append(count[m][-1] + (1 if v == m else 0))

            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 8))

            for m in count.keys():
                ax.plot(list(range(len(count[m]))), count[m], label=m)
            plt.legend(loc='upper left')
            plt.xlabel('tours')
            plt.ylabel("Utilisations cumulées de l'operateur")
            if s['parameters']['mutation'] == ['1-flip']:
                plt.title('1-flip')
                # plt.savefig('1-flip' + ".png", pad_inches=0, bbox_inches='tight')
            else:
                plt.title(s['parameters']['mutation'][1] + str(i))
                plt.savefig(s['parameters']['mutation'][1] + ".png", pad_inches=0, bbox_inches='tight')

            plt.show()
            # from algo_gen.tools.plot import show_stats
            #
            # if s['parameters']['mutation'] == ['1-flip']:
            #     show_stats(s, '1-flip')
            # else:
            #     show_stats(s,s['parameters']['mutation'][1])
            #
