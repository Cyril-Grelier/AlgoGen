# import copy
# import time
# from itertools import product
#
# from joblib import Parallel, delayed
#
# from algo_gen.classes import Population
# from algo_gen.tools.tools import save_stats_parameters
#
#
# def iterate(a):
#     for p in a:
#         # Always sort the keys of a dictionary, for reproducibility
#         items = sorted(p.items())
#         if not items:
#             yield {}
#         else:
#             keys, values = zip(*items)
#             for v in product(*values):
#                 params = dict(zip(keys, v))
#                 yield params
#
#
# def gridsearch(base_parameter, dict_changements):
#     result = []
#     nb_tests = 0
#     for _ in iterate([dict_changements]):
#         nb_tests += 1
#
#     for i, currents_changements in enumerate(iterate([dict_changements])):
#         print(f'\r{i}/{nb_tests} : {i * 100 / nb_tests:2.2f} -- {currents_changements}' + ' ' * 30, end='')
#         for k, v in currents_changements.items():
#             base_parameter[k] = v
#         # base_parameter['stop after no change'] = int(base_parameter['nb turn max'] * 0.10)
#
#         population = Population(base_parameter)
#         stats = population.start()
#         result.append((copy.deepcopy(dict_changements), stats))
#     return result
#
#
# def gridsearch_multipross(base_parameter, dict_changements):
#     backend = 'multiprocessing'
#     Parallel(n_jobs=8, backend=backend)(delayed(
#         launcher_for_gridsearch_mp)(copy.deepcopy(base_parameter), currents_changements, i)
#                                         for i, currents_changements in enumerate(iterate([dict_changements])))
#
#
# def launcher_for_gridsearch_mp(base, changements, file_name):
#     for k, i in changements.items():
#         base[k] = i
#     # base['stop after no change'] = int(base['nb turn max'] * 0.10)
#
#     population = Population(base)
#     start = time.time()
#     stats = population.start()
#     stop = time.time()
#     stats['time'] = float(stop - start)
#     save_stats_parameters(stats, file_name)
#
#
# if __name__ == '__main__':
#     parameter = {
#         'configuration name': 'config1',
#         'individual': ['onemax', 'IndividualOneMax'],
#
#         'population size': 50,  # 100 200 500
#         'chromosome size': 20,  # 5 10 50 100
#
#         'nb turn max': 500,
#         'stop after no change': 50,  # int(config['nb turn max']*0.10)
#
#         'selection': 'select_tournament',  # 'select_random' 'select_best' 'select_tournament'
#         'proportion selection': 0.04,  # 2 / config['population size']
#         'nb selected tournament': 20,  # int(config['population size']*0.4)
#
#         'proportion crossover': 1,
#         'type crossover': 'uniforme',  # 'mono-point' 'uniforme'
#
#         'mutation': ['n-flip', 3],
#         # ['n-flip', 1] ['n-flip', 3] ['n-flip', 5] ['bit-flip', 0.2] ['bit-flip', 0.5] ['bit-flip', 0.8]
#         'proportion mutation': 0.2,  # 0.1 0.2 0.5 0.8
#
#         'insertion': 'age',  # 'age' 'fitness'
#     }
#     changement = {
#         'population size': [50, 100, 500],
#         'chromosome size': [10, 50, 100],
#     }
#
#     gridsearch_multipross(parameter, changement)
#
# # from gridsearch import iterate
# #
# # changements = {
# #     #     'population size': [ 50, 100, 500],
# #     #     'chromosome size': [10, 50, 100],
# #
# #     'selection': ['select_random', 'select_best', 'select_tournament'],
# #     #     'proportion selection': [0.04, 0.08, 0.16],
# #
# #     'proportion crossover': [0, 0.5, 1],
# #     'type crossover': ['mono-point', 'uniforme'],
# #
# #     'mutation': [['n-flip', 1], ['n-flip', 3], ['n-flip', 5], ['bit-flip']],
# #     'proportion mutation': [0.1, 0.2, 0.5, 0.8],
# #
# #     'insertion': ['age', 'fitness'],
# # }
# # i = 0
# # for a in iterate([changements]):
# #     i += 1
# # print(i)
#
# # %%time
# # genetic_algorithm.gridsearch_multipross(parameters, changements)
#
# # import json
# # import glob
# #
# #
# # results = []
# # for f in glob.glob('results/*.json'):
# #     with open(f, 'r') as r:
# #         for l in r.readlines():
# #             x = json.loads(l)
# #             results.append(x)
# #
# # results.sort(key=lambda x: x['min_fitness'][-1])
# #
# # show_stats(results[0])
# # show_stats(results[1])
# # show_stats(results[2])
# # show_stats(results[-1])
# # show_stats(results[-2])
# # show_stats(results[-3])
