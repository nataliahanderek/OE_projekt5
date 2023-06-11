import numpy as np
from mealpy.swarm_based.MFO import OriginalMFO

# nasza funkcja celu, wzięłam tą co we wcześniejszych projektach - Booth
# przypomnienie: wartość funkcji celu powinna być bliska 0, a wartości zmiennych bliskie [1, 3]
def booth_function(x):
    return (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2

# argumenty dla MFO jako słownik
problem_dict = {
    "fit_func": booth_function,
    "lb": [-10, -10],  # Dolne ograniczenia zmiennych
    "ub": [10, 10],  # Górne ograniczenia zmiennych
    "minmax": "min",
}

# różne konfiguracje do testowania
configurations = [
    {"epoch": 700, "pop_size": 350}
    # {"epoch": 200, "pop_size": 100}
    # {"epoch": 1000, "pop_size": 500}
]

for config in configurations:
    epoch = config["epoch"]
    pop_size = config["pop_size"]

    model = OriginalMFO(epoch, pop_size)
    best_position, best_fitness = model.solve(problem_dict)

    history = model.history
    positions = history.list_current_best
    worst = history.list_current_worst
    # model.history.save_diversity_chart(filename="diversity")
    model.history.save_global_best_fitness_chart(filename="best")

    print("\n")
    print("Konfiguracja: epoch = ", epoch, ", pop_size = ", pop_size)
    print("Najlepsze rozwiązanie znalezione przez MFO:")
    print("Wartość funkcji celu:", best_fitness)
    print("Wartości zmiennych:", best_position)
    print("\n")