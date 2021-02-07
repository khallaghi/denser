from pyeasyga.pyeasyga import GeneticAlgorithm
from genetic_algorithm.convolution import Convolution
from genetic_algorithm.max_pooling import MaxPooling
from genetic_algorithm.flatten import Flatten
from genetic_algorithm.classification import Classification
from network_builder.network_evaluator import get_network_fitness
import copy
import random

all_accuracies = []
all_losses = []


def create_individual(data):
    feature_size = random.randint(1, 10)
    classification_size = random.randint(1, 10)
    _individual = []
    for i in range(feature_size):
        if random.randint(0, 1) == 1:
            _individual.append(Convolution())
        else:
            _individual.append(MaxPooling())
    for i in range(classification_size):
        _individual.append(Classification())
    return _individual


def crossover(parent_1, parent_2):
    border_idx_1 = find_module_border(parent_1)
    border_idx_2 = find_module_border(parent_2)
    if random.randint(0, 1) == 0:
        rand_index_1 = random.randrange(border_idx_1 + 1)
        rand_index_2 = random.randrange(border_idx_2 + 1)
    else:
        rand_index_1 = random.randrange(border_idx_1 + 1, len(parent_1))
        rand_index_2 = random.randrange(border_idx_2 + 1, len(parent_2))

    child_1 = parent_1[:rand_index_1 + 1] + parent_2[rand_index_2:]
    child_2 = parent_2[:rand_index_2 + 1] + parent_1[rand_index_1:]
    return child_1, child_2


def mutate(individual):
    random_idx = random.randint(0, 2)
    mutate_index = random.randrange(len(individual))
    if random_idx == 0:
        # remove
        individual.pop(mutate_index)
    elif random_idx == 1:
        # copy
        individual.insert(mutate_index+1, individual[mutate_index])
    elif random_idx == 2:
        # add
        if type(individual[mutate_index]).__name__ == Classification.__name__:
            individual.insert(mutate_index + 1, Classification())
        else:
            if random.randint(0, 1) == 0:
                individual.insert(mutate_index + 1, MaxPooling())
            else:
                individual.insert(mutate_index + 1, Convolution())


def selection(population):
    return random.choice(population)


def find_module_border(individual):
    for i in range(len(individual)):
        if type(individual[i]).__name__ == Convolution.__name__ or type(individual[i]).__name__ == MaxPooling.__name__:
            if type(individual[i + 1]).__name__ == Classification.__name__:
                return i


def fitness(individual, data):

    # insert flatten
    individual_copy = copy.deepcopy(individual)
    individual_copy.insert(0, Convolution(input_shape=(32, 32, 3)))
    individual_copy.append(Classification(net_size=10, activation=None))
    border_idx = find_module_border(individual_copy)
    individual_copy.insert(border_idx + 1, Flatten())
    # print
    rules = []
    for item in individual_copy:
        rules.append(str(item))
    final_network = "\n".join(rules)
    print(final_network)

    # do the math
    _fitness, loss = get_network_fitness(final_network)
    all_accuracies.append(_fitness)
    all_losses.append(loss)
    print('=========================================================')
    return _fitness


data = [["conv", "pool"], "flatten", "dense"]
ga = GeneticAlgorithm(data, 20, 50, 0.8, 0.2, True, True)
ga.create_individual = create_individual
ga.crossover_function = crossover
ga.mutate_function = mutate
ga.selection_function = selection
ga.fitness_function = fitness

