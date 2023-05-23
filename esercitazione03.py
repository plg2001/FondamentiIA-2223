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
    n = len(individual)
    coppie_totali = n*(n-1)/2
    coppie_cattive = 0

    for i in range(0,n):
        for j in range(i+1,n):
            if(individual[i] == individual[j]):
                 coppie_cattive +=1
            if((abs(individual[i] - individual[j]) == abs(i-j)) or (abs(individual[i]+ individual[j]) == abs(i+j))):
                coppie_cattive +=1
    
    
                
    return coppie_totali - coppie_cattive
    


def roulette_wheel_selection(population, fitnesses, k):
    """Implementare la fase di selezione dei genitori.

    Dati:
    - population: lista di individui
    - fitnesses: lista dei valori di fitness per ogni individuo
    - k: numero di parent da selezionare
    ritornare una lista di k individui, estratti a caso dalla
    popolazione con probabilità proporzionale al loro valore di fitness.
    """

    parents = []
    fitness_sum = sum(fitnesses)

    for i in range(k):
        # Seleziona un numero casuale compreso tra 0 e la somma dei valori di fitness
        selection = random.uniform(0, fitness_sum)

        # Itera sui candidati fino a trovare quello selezionato
        fitness_cumulative = 0
        for j in range(len(population)):
            fitness_cumulative += fitnesses[j]
            if fitness_cumulative > selection:
                # Aggiungi il candidato selezionato alla lista dei genitori
                parents.append(population[j])
                break

    return parents

    


def mutation(individual, rate):
    """Implementare la fase di mutazione.

    Dato un individuo e la probabilità di mutazione, ritornare *una
    copia* dell'individuo con applicata l'eventuale mutazione (i.e. un
    singolo gene ri-estratto tra i valori possibili, vedi slides). Si
    ricorda che, in python, data una lista l se ne può ottenere una
    copia con l.copy().
    """
    # Crea una copia dell'individuo
    mutated_individual = individual.copy()

    # Applica l'eventuale mutazione a ciascun gene
    for i in range(len(mutated_individual)):
        if random.random() < rate:
            # Se il valore generato casualmente è minore della probabilità di mutazione, muta il gene
            mutated_individual[i] = random.choice(range(1, len(mutated_individual) + 1))

    return mutated_individual


def crossover(p, q):
    """Implementare la fase di crossover.

    Dati due individui genitori p, q, ritornare una *nuova* coppia di
    individui figli che siano il risultato del crossover tra p e q
    (vedi slides).
    """
    # Crea due nuovi individui figli inizialmente vuoti
    child1 = []
    child2 = []

    # Scegli casualmente il punto di crossover
    n = len(p)
    cross_point = random.randint(0, n-1)

    # Genera i due figli attraverso il crossover
    child1 = p[:cross_point] + q[cross_point:]
    child2 = q[:cross_point] + p[cross_point:]

    print(child1,child2)
    return (child1, child2)

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