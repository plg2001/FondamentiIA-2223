import itertools
import math
import random


# Si ricorda che python mette a disposizione nel modulo random varie
# funzioni che sono necessarie per implementare le fasi stocastiche
# dell'algoritmo genetico. Per maggiori dettagli consultare
# https://docs.python.org/3/library/random.html


# Esercizio 1

def fitness_fn(individual):
    """Implementare la funzione di fitness.

    Dato un individuo, ritornare il valore di fitness corrispondente
    (vedi slides).
    """
    pass

def roulette_wheel_selection(population, fitnesses, k):
    """Implementare la fase di selezione dei genitori.

    Dati:
    - population: lista di individui
    - fitnesses: lista dei valori di fitness per ogni individuo
    - k: numero di parent da selezionare
    ritornare una lista di k individui, estratti a caso dalla
    popolazione con probabilità proporzionale al loro valore di fitness.
    """
    pass

def mutation(individual, rate):
    """Implementare la fase di mutazione.

    Dato un individuo e la probabilità di mutazione, ritornare *una
    copia* dell'individuo con applicata l'eventuale mutazione (i.e. un
    singolo gene ri-estratto tra i valori possibili, vedi slides). Si
    ricorda che, in python, data una lista l se ne può ottenere una
    copia con l.copy().
    """
    pass

def crossover(p, q):
    """Implementare la fase di crossover.

    Dati due individui genitori p, q, ritornare una *nuova* coppia di
    individui figli che siano il risultato del crossover tra p e q
    (vedi slides).
    """
    pass

# Esercizio 2

def success_rate(**parameters):
    """Implementare la stima del success rate con esperimenti ripetuti.

    Dati i parametri dell'algoritmo genetico da passare a
    run_ga_n_queens (nel dizionario parameters), ripetere diversi
    esperimenti (almeno 25) per stimare e ritornare il success rate
    dell'algoritmo genetico.

    Si ricorda che per passare i parametri in maniera compatta alla
    funzione run_ga_n_queens si può usare la sintassi python
    run_ga_n_queens(**parameters).
    """
    pass


# Implementation

def random_individual(n):
    return [random.randrange(n) for _ in range(n)]

def genetic_algorithm_step(
    population,
    fitnesses,
    mutation_rate,
):
    parents = roulette_wheel_selection(population, fitnesses, k=len(population))
    offspring = []
    for x, y in zip(parents[::2], parents[1::2]):
        offspring += crossover(x, y)
    offspring = [mutation(x, rate=mutation_rate) for x in offspring]
    offspring_fitnesses = [fitness_fn(x) for x in offspring]
    population, fitnesses = offspring, offspring_fitnesses
    return population, fitnesses

def run_ga_n_queens(
    n,
    population,
    mutation_rate,
    steps,
    verbose=False
):
    target_fitness = int(n*(n-1)/2)
    ga_params = dict(
        mutation_rate=mutation_rate,
    )
    fitnesses = [fitness_fn(x) for x in population]
    solved = False
    if verbose:
        print(f'[0] Champion fitness: {max(fitnesses)}; population fitness (mean +- std): {mean(fitnesses)} +- {std(fitnesses)}')
    for i in range(steps):
        population, fitnesses = genetic_algorithm_step(population, fitnesses, **ga_params)
        if verbose:
            print(f'[{i+1}] Champion fitness: {max(fitnesses)}; population fitness (mean +- std): {mean(fitnesses)} +- {std(fitnesses)}')
        if max(fitnesses) == target_fitness:
            if verbose:
                print(f'Solution found at generation {i+1}.')
            solved = True
            break
    else:
        if verbose:
            print(f'Solution not found in the given number of generations.')
    i_champ = argmax(fitnesses)
    champion = population[i_champ]
    if verbose:
        print(f'Champion individual (fitness: {fitnesses[i_champ]}):')
        print_board(champion)
    return solved

def argmax(v):
    i_m, m = None, None
    for i, vi in enumerate(v):
        if i_m is None or vi > m:
            m = vi
            i_m = i
    return i_m

def mean(v):
    return sum(v) / len(v)

def std(v):
    mu = mean(v)
    return math.sqrt(sum([(x - mu)**2 for x in v]) / len(v))

def print_board(individual):
    print(f'Board resulting from {individual}')
    n = len(individual)
    board = [["-" for _ in range(n)] for _ in range(n)]
    for j, i in enumerate(individual):
        board[i][j] = "Q"
    for i in range(n):
        for j in range(n):
            k = board[i][j]
            print(f'{k} ', end="")
        print()


def main():
    random.seed(12345)
    pop_size = 50
    population = [random_individual(n=6) for _ in range(pop_size)]
    params = dict(
        mutation_rate=0.5,
    )

    print(f'Esercizio 1:')
    print(f'------------')
    run_ga_n_queens(
        n=6,
        population=population,
        **params,
        steps=500,
        verbose=True
    )

    print(f'\nEsercizio 2:')
    print(f'------------')
    for n in [4, 5, 6, 7, 8]:
        population = [random_individual(n=n) for _ in range(pop_size)]
        r = success_rate(population=population, n=n, steps=500, **params)
        print(f'Success rate for the {n}-queens problems: {r}')


if __name__ == '__main__':
    main()