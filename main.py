import sys
import numpy as np
from random import SystemRandom
import matplotlib.pyplot as plt

class Generation(object):
    def __init__(self, best_candidate, best_val, average):
        self.best_candidate = best_candidate
        self.best_value = best_val
        self.average = average

class limits(object):
    def __init__(self, x_min, x_max, y_min, y_max):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

population_size = [20, 50, 100, 200]
num_of_gens = [50, 100, 200]
egg_limits = limits(-512, 512, -512, 512)
holder_limits = limits(-10, 10, -10, 10)
K = 0.5
Cr = 0.8
graph_store = {}

def egg_holder_function(candidate):
    x = candidate[0]
    y = candidate[1]
    return -(y+47)*np.sin(np.sqrt(np.abs(x/2+(y+47))))-x*np.sin(np.sqrt(np.abs(x-(y+47))))

def holder_table_function(candidate):
    x = candidate[0]
    y = candidate[1]
    return -np.abs(np.sin(x)*np.cos(y)*np.exp(np.abs(1-(np.sqrt(x**2+y**2)/np.pi))))

def initialization(n, limits):
    initial_vectors = []
    for i in range(n):
        x_rand = SystemRandom().uniform(limits.x_min, limits.x_max + sys.float_info.epsilon)
        y_rand = SystemRandom().uniform(limits.y_min, limits.y_max + sys.float_info.epsilon)
        initial_vectors.append(np.array([x_rand, y_rand])) 
    return initial_vectors

def get_F(_min=-2, _max=2):
    return SystemRandom().uniform(_min, _max + sys.float_info.epsilon)

def check_limits(z, limits):
    x = z[0]
    y = z[1]

    if limits.x_min <= x <= limits.x_max and limits.y_min <= y <= limits.y_max:
        return True
    else:
        return False

def elitism(function, parent_vectors, trial_vectors):
    selected_vectors = []
    _best_value = sys.maxint
    _average = 0

    for parent, trial in zip(parent_vectors, trial_vectors):
        
        parent_val = function(parent)
        trial_val = function(trial)

        if parent_val < trial_val:
            selected = parent
            _average += parent_val
            if parent_val < _best_value:
                _best_value = parent_val
                _best_candidate = selected
                
        else:
            selected = trial
            _average += trial_val
            if trial_val < _best_value:
                _best_value = trial_val
                _best_candidate = selected
        
        selected_vectors.append(selected)
    _average /= len(trial_vectors)
    return selected_vectors, Generation(_best_candidate, _best_value, _average)

def plot(function_name, generations, pop_size, gens_count):
    global graph_store
    x = [i for i in range(len(generations))]
    y_best = [i.best_value for i in generations]
    y_avg = [i.average for i in generations]
    graph_store[pop_size] = {'y_best' : y_best, 'y_avg': y_avg}

    if(len(graph_store) == len(population_size)):
        n = len(population_size)/2
        plt.suptitle(function_name.title()+"\n#Generations: {}".format(gens_count), fontsize=16)
        for index, pop_size in enumerate(population_size):
            plt.subplot(n, n, index+1)
            plt.title("Population size: {}".format(pop_size))
            plt.plot(x, graph_store[pop_size]['y_avg'], label='Average value')
            plt.legend()
            plt.plot(x, graph_store[pop_size]['y_best'], label='Best value')
            plt.legend()
            plt.xlabel('Number of generations -- >')
            plt.ylabel('Function value -->')
            plt.legend()
        graph_store = {}
        plt.show()
    
def DE(function, limits, pop_size, gens_count):
    
    parent_vectors = initialization(pop_size, limits)
    best_of_all_generations = []
    no_of_gens = gens_count
    function_name = function.__name__.title()
    function_name = function_name.replace('_', ' ')
    while(gens_count-1):
        trial_vectors = []
        F = get_F()
        gens_count -= 1
        for index, candidate in enumerate(parent_vectors):            
            while(True):
            	parents_minus_i = parent_vectors[:]
            	parents_minus_i.pop(index)
                r1, r2, r3 = SystemRandom().sample(parents_minus_i, 3)
                mutant = candidate + K*(r1 - candidate) + F*(r2 - r3)
                
                z = [None for i in range(len(mutant))]
                for j in range(len(mutant)):
                    Crp = SystemRandom().random()
                    if Crp <= Cr:
                        z[j] = mutant[j]
                    else:
                        z[j] = candidate[j]      
                if(check_limits(z, limits)):
                    break
            trial_vectors.append(np.array(z))
        new_gen, best_of_generation = elitism(function, parent_vectors, trial_vectors)
        best_of_all_generations.append(best_of_generation)
        parent_vectors = new_gen[:]
    print 'Function: {}\n#Generations: {}\nPopulation size: {}\nBest value: {}\nBest candidate: {}\n'.format(function_name, no_of_gens, pop_size, best_of_generation.best_value, best_of_generation.best_candidate)

    plot(function_name, best_of_all_generations, pop_size, no_of_gens)
    
if __name__ == "__main__":
    functions = [{'function': egg_holder_function, 'limits': egg_limits},
                 {'function': holder_table_function, 'limits': holder_limits}]
    for func in functions:
        for gen in num_of_gens:
            for pop in population_size:
            	DE(func['function'], func['limits'], pop, gen)
