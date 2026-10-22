import numpy as np

class AVLO:
    """
    African Vulture and Lemur Optimizer (AVLO)
    A hybrid heuristic algorithm for hyperparameter tuning.
    """
    def __init__(self, objective_function, bounds, pop_size=10, max_iter=25):
        self.obj_func = objective_function
        self.bounds = np.array(bounds)
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.dim = len(bounds)
        
        # Initialize population
        self.population = np.zeros((self.pop_size, self.dim))
        for i in range(self.pop_size):
            for j in range(self.dim):
                self.population[i, j] = np.random.uniform(self.bounds[j, 0], self.bounds[j, 1])
                
        self.fitness = np.zeros(self.pop_size)
        self.best_sol = None
        self.best_fitness = float('inf')

    def optimize(self):
        # Initial evaluation
        for i in range(self.pop_size):
            self.fitness[i] = self.obj_func(self.population[i])
            if self.fitness[i] < self.best_fitness:
                self.best_fitness = self.fitness[i]
                self.best_sol = np.copy(self.population[i])
                
        for iteration in range(self.max_iter):
            for i in range(self.pop_size):
                # AVOA update strategy (exploration)
                ps1 = self.best_sol + np.random.randn(self.dim) * (self.bounds[:, 1] - self.bounds[:, 0]) * 0.1
                
                # Lemur Optimizer update strategy (exploitation)
                rnd = np.random.rand()
                if rnd < 0.5:
                    ps2 = self.population[i] + np.random.rand(self.dim) * (self.best_sol - self.population[i])
                else:
                    idx = np.random.randint(0, self.pop_size)
                    ps2 = self.population[i] + np.random.rand(self.dim) * (self.population[idx] - self.population[i])
                
                # Hybrid AVLO update (Eq 1 from paper)
                # ps = mean(ps1, ps2) + bs/wf
                # Simplified representation:
                new_pos = (ps1 + ps2) / 2.0
                
                # Bound checking
                new_pos = np.clip(new_pos, self.bounds[:, 0], self.bounds[:, 1])
                
                # Evaluate
                new_fit = self.obj_func(new_pos)
                
                if new_fit < self.fitness[i]:
                    self.fitness[i] = new_fit
                    self.population[i] = new_pos
                    
                if new_fit < self.best_fitness:
                    self.best_fitness = new_fit
                    self.best_sol = np.copy(new_pos)
                    
            print(f"Iteration {iteration+1}/{self.max_iter}, Best Fitness: {self.best_fitness:.4f}")
            
        return self.best_sol, self.best_fitness
