import json


def get_import(parameters, name):
    imp, func = parameters[name]
    function = getattr(__import__(imp, fromlist=[func]), func)
    return function


def load_parameters(config_name):
    with open('configurations/' + config_name + ".json") as c:
        return json.load(c)


def save_stats_parameters(stats, file_name):
    with open(f'results/{file_name}.json', 'a') as results:
        json.dump(stats, results)
        results.write("\n")
