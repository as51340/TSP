# System dependencies
import numpy as np
import random
import time
import math
import sys
from copy import deepcopy
# User dependencies
from Reporter import Reporter


class Individual:

	def __init__(self, num_cities=0, order=None, alpha=None, flip_rate=None):
		"""
        :param fun: function to evaluate individual
        :param num_cities: Size of the individual
        :param alpha: Mutation rate
        :param max_iterations maximum number of iterations for performing local search operator
        """
		if order is None:
			if num_cities == 0:
				sys.exit("Given 0 cities, cannot create random configuration, finishing program...")
			self.order = np.random.choice(np.arange(0, num_cities, dtype=np.uint32), replace=False, size=num_cities)
		else:
			self.order = order

		self.num_cities = self.order.shape[0]
		if alpha is None:
			self.alpha = max(0.05, 0.15 + 0.05 * random.gauss(0.0, 1.0))
		else:
			self.alpha = alpha
		if flip_rate is None:
			self.flip_rate = max(0.1, 0.5 + 0.05 * random.gauss(0.0, 1.0))
		else:
			self.flip_rate = flip_rate
		# When new individual is created compute its edges
		self.edges = None

	def get_edges(self):
		if self.edges is None:
			self.edges = self.compute_all_edges()
		return self.edges

	def neigh_edges(self, other):
		"""
        Calculates function of neighbourhood function by looking number of different edges between them.
        :param other: Individual
        :return:
        """
		return np.setdiff1d(self.get_edges(), other.get_edges()).shape[0]

	def get_edges_as_dict(self):
		return dict((self.order[city_ind - 1], self.order[city_ind]) for city_ind in range(0, self.num_cities))

	def neigh_edges_pop(self, pop):
		"""
        Calculate neighbouring edges with all individuals within population
        :param pop:
        :return:
        """
		distances = np.zeros(pop.shape[0])
		for i, curr_ind in enumerate(pop):
			distances[i] = self.neigh_edges(curr_ind)
		return distances

	def compute_all_edges(self):
		"""
        Computes all edges for current individual.
        :return:
        """
		edges = np.zeros(self.num_cities, dtype=np.uint32)
		for i in range(self.num_cities - 1):
			edges[i] = self.order[i] * 1000 + self.order[i + 1]
		edges[self.num_cities - 1] = self.order[self.num_cities - 1] * 1000 + self.order[0]
		self.edges = edges
		return edges

	def compute_edges(self, order):
		"""
        Based on idea that there will be at most 1000 cities. Doesn't count loop - used in some mutation process.
        :return:
        """
		n = order.shape[0]
		edges = np.zeros(n - 1, dtype=np.uint32)
		for i in range(n - 1):
			edges[i] = order[i] * 1000 + order[i + 1]
		return edges


class Parameters:

    def __init__(self, lambdaa, mu, iterations, alpha, k, mutation_rate_coeff, sigma_sharing, alpha_sharing, window_size_start=2, window_size_max=10):
        """
        :param lambdaa: Population size
        :param mu: offspring size
        :param iterations: number of iterations algorithm will be run
        :param alpha for sigma selection
        :param k: k tournament selection
        :param mutation_rate_coeff: you are going to multiply all mutation rates if you realize they are
        :param window_size_start for local search operator what is the initial space
        :param window_size_max: what is the maximum window size for local search operator
        below some threshold
        """
        self.lambdaa = lambdaa
        self.mu = mu
        self.iterations = iterations
        self.alpha = alpha
        self.k = k
        self.mutation_rate_coeff = mutation_rate_coeff
        self.sigma_sharing = sigma_sharing  # sigma distance for neighbourhood - fitness sharing di
        self.alpha_sharing = alpha_sharing  # alpha coefficient for fitness sharing diversity promotion
        self.window_size_start = window_size_start
        self.window_size_max = window_size_max

    def __str__(self):
        return f"Lambdaa: {self.lambdaa}, Mu: {self.mu}, Iterations: {self.iterations}, Alpha: {self.alpha}\n"


class EdgeMap:

    def __init__(self, p1, p2):
        self.num_cities = p1.shape[0]
        self.edge_map = self.construct_edge_map(p1, p2)

    def construct_edge_map(self, p1, p2):
        """
        Constructs edge map from given two parents, for genetic edge crossover operator.
        :param p1:
        :param p2:
        :return:
        """
        # Should be of size self.num_cities
        edge_map = dict()
        for i in range(self.num_cities):
            if edge_map.get(p1[i]) is None:
                edge_map[p1[i]] = set()
            if edge_map.get(p2[i]) is None:
                edge_map[p2[i]] = set()
            if i > 0:  # Otherwise don't append -1
                edge_map[p1[i]].add(p1[i - 1])
                edge_map[p2[i]].add(p2[i - 1])
            if i != self.num_cities - 1:  # if it's not last element
                edge_map[p1[i]].add(p1[i + 1])
                edge_map[p2[i]].add(p2[i + 1])
        # Append last connections
        edge_map[p1[0]].add(p1[self.num_cities - 1])
        edge_map[p2[0]].add(p2[self.num_cities - 1])
        edge_map[p1[self.num_cities - 1]].add(p1[0])
        edge_map[p2[self.num_cities - 1]].add(p2[0])
        return edge_map

    def find_city_fewest_edges(self, cities):
        """
        Finds city which has fewest edges from given list of cities.
        :return:
        """
        min_num_edges, current_city = self.num_cities+1, 0
        for city in cities:
            edges_len = len(self.edge_map[city])
            if edges_len < min_num_edges:
                min_num_edges = edges_len
                current_city = city
        return current_city

    def update_edges_map(self, current_city):
        """
        Removes all occurences of current city from edges list
        :param current_city:
        :return:
        """
        for city, edges in self.edge_map.items():
            if current_city in edges:
                edges.remove(current_city)

    def delete_city(self, city):
        """
        Deletes city
        :param city:
        :return:
        """
        del self.edge_map[city]

    def get_edges(self, city):
        """
        Returns edges for given city
        :param city:
        :return:
        """
        return self.edge_map[city]

    def get_random_city(self):
        """
        Selects some of random remaining cities.
        :return:
        """
        return random.choice(list(self.edge_map.keys()))


class r0876363:

	def __init__(self):
		"""
		Reference to parameters used
		:param parameters:
		"""
		# This is to be convenient with their code
		self.reporter = Reporter(self.__class__.__name__)
		self.parameters = Parameters(200, 600, 2000, 0.9, 10, 1.1, 15, 1.05)
		self.weights = None  # will be initialized later
		self.ordered_weights_indices = None  # also initialized later
		self.num_cities = None  # also later
		self.population = None  # also later
		self.objective_values = None  # also later
		# For distinguishing several algorithms
		self.settings = np.array([0, 1, 2, 0, 0, 1, 0])
		# Just indices - we want to reuse them multiple times
		self.lambda_indices = np.arange(0, self.parameters.lambdaa, dtype=np.uint32)
		# self.lso_2opt_time = 0.0
		# self.lso_precise_time = 0.0
		# self.inverse_mutation_time = 0.0
		# self.scramble_mutation_time = 0.0
		# self.insert_mutation_time = 0.0
		# self.rnd_mutation_time = 0.0
		# self.total_eval_time = 0.0
		# self.crossover_time = 0.0
		# self.loop_time = 0.0
		# self.elimination_time = 0.0
		# self.e_time2 = 0.0
		# self.e_time3 = 0.0
		self.lso_precise_iteration_time = 0.0
		self.iteration_time = 0.0
		# self.fitness_sharing_iteration_time = 0.0

	# Evaluation

	def get_objective_values(self, population):
		"""
		Returns objective values from population without calculating them.
		:param population:
		:return:
		"""
		obj_values = np.zeros(population.shape[0])
		for i, ind in enumerate(population):
			obj_values[i] = ind.obj_value
		return obj_values

	def objf(self, order):
		sum = 0
		for i in range(self.num_cities-1):
			sum += self.weights[order[i], order[i + 1]]
		return sum + self.weights[order[order.shape[0] - 1], order[0]]

	def obj(self, candidate):
		loc = sum(self.weights[candidate.order[i], candidate.order[i+1]] for i in range(self.num_cities-1))
		loc += self.weights[candidate.order[self.num_cities - 1], candidate.order[0]]
		return loc

	def objfpop(self, candidates):
		array = np.zeros(candidates.shape[0])
		for i in range(candidates.shape[0]):
			array[i] = self.objf(candidates[i].order)
		return array

	# Initialization

	def initialize_randomly(self):
		"""
		Return population. Now it's very simple random initialization.
		:return:
		"""
		self.population = np.empty(self.parameters.lambdaa, dtype=Individual)
		self.objective_values = np.zeros(self.parameters.lambdaa, dtype=np.double)
		for i in range(self.parameters.lambdaa):
			self.population[i] = Individual(num_cities=self.num_cities)
			self.objective_values[i] = self.objf(self.population[i].order)
		return self.population

	def heur_init_ind(self, start_city):
		"""
		Create individual heuristically but not all nodes just a part of them.
		:param individual:
		:return:
		"""
		exp = int(np.random.exponential(3))
		n = max(self.num_cities - exp, 1)
		# print(f"Exp: {exp} n: {n} start_city: {start_city} num_cities: {self.num_cities}")
		order_ind_heur = np.empty(n, dtype=np.int32)
		order_ind_heur[0] = start_city

		used_cities = {start_city}  # careful because if empty curly braces it is dict
		for i in range(1, n):
			ind = 1
			while self.ordered_weights_indices[order_ind_heur[i-1], ind] in used_cities:
				ind += 1
			order_ind_heur[i] = self.ordered_weights_indices[order_ind_heur[i-1], ind]
			used_cities.add(order_ind_heur[i])

		all = np.arange(self.num_cities)
		rnd_part = np.random.permutation(np.setdiff1d(all, order_ind_heur))
		fin_order = np.concatenate((order_ind_heur, rnd_part))
		return Individual(order=fin_order)

	def heur_init(self):
		# Needs better way - too expensive operation
		self.population = np.empty(self.parameters.lambdaa, dtype=Individual)
		self.objective_values = np.zeros(self.parameters.lambdaa, dtype=np.double)
		max_lambdaa_cities = max(self.parameters.lambdaa, self.num_cities)
		heuristic_candidates = np.empty(max_lambdaa_cities, dtype=Individual)
		heuristic_candidates_obj_values = np.zeros(max_lambdaa_cities)
		st = time.time()
		for i in range(max_lambdaa_cities):
			x = None
			if i >= self.num_cities:
				x = np.random.randint(low=0, high=self.num_cities)
			else:
				x = i
			heuristic_candidates[i] = self.heur_init_ind(x)
			heuristic_candidates_obj_values[i] = self.objf(heuristic_candidates[i].order)
		# print(f"All heuristic time took: {time.time()-st:.2f}")

		sort_indices = np.argsort(heuristic_candidates_obj_values)
		heuristic_candidates_sorted = np.take(heuristic_candidates, sort_indices)
		heuristic_candidates_obj_values_sorted = np.take(heuristic_candidates_obj_values, sort_indices)

		self.population[0] = heuristic_candidates_sorted[0]
		self.objective_values[0] = heuristic_candidates_obj_values_sorted[0]

		sample_size = 5

		for i in range(1, self.parameters.lambdaa):
			# Choose randomly other individuals with which you will compare your own Individual
			tournament_players_indices = np.random.choice(max_lambdaa_cities, size=sample_size, replace=False)  # select players indices
			neighbours_lengths = np.zeros(sample_size, dtype=np.uint32)
			# sum of all neighbours: Between each candidate and population up to i
			for j in range(sample_size):
				neighbours_lengths[j] = sum(heuristic_candidates_sorted[tournament_players_indices[j]].neigh_edges(self.population[k]) for k in range(i)) / i

			max_index = np.argmax(neighbours_lengths)
			self.population[i] = deepcopy(heuristic_candidates_sorted[tournament_players_indices[max_index]])
			self.objective_values[i] = heuristic_candidates_obj_values_sorted[tournament_players_indices[max_index]]

		return self.population

	def partial_population_init(self):
		self.population = np.empty(self.parameters.lambdaa, dtype=Individual)
		self.objective_values = np.zeros(self.parameters.lambdaa, dtype=np.double)
		max_lambdaa_cities = max(self.parameters.lambdaa, self.num_cities)
		heuristic_candidates = np.empty(max_lambdaa_cities, dtype=Individual)
		heuristic_candidates_obj_values = np.zeros(max_lambdaa_cities)
		st = time.time()
		for i in range(max_lambdaa_cities):
			x = None
			if i >= self.num_cities:
				x = np.random.randint(low=0, high=self.num_cities)
			else:
				x = i
			heuristic_candidates[i] = self.heur_init_ind(x)
			heuristic_candidates_obj_values[i] = self.objf(heuristic_candidates[i].order)
		# print(f"All heuristic time took: {time.time() - st:.2f}")

		sort_indices = np.argsort(heuristic_candidates_obj_values)
		heuristic_candidates_sorted = np.take(heuristic_candidates, sort_indices)
		heuristic_candidates_obj_values_sorted = np.take(heuristic_candidates_obj_values, sort_indices)

		self.population[0] = heuristic_candidates_sorted[0]
		self.objective_values[0] = heuristic_candidates_obj_values_sorted[0]

		sample_size = 5

		random_init_rates = np.random.random(self.parameters.lambdaa)
		heur_rnd_rate = 0.5
		for i in range(1, self.parameters.lambdaa):
			if random_init_rates[i] > heur_rnd_rate:
				self.population[i] = Individual(num_cities=self.num_cities)
				self.objective_values[i] = self.objf(self.population[i].order)
			else:
				# Choose randomly other individuals with which you will compare your own Individual
				tournament_players_indices = np.random.choice(max_lambdaa_cities, size=sample_size,
															  replace=False)  # select players indices
				neighbours_lengths = np.zeros(sample_size, dtype=np.uint32)
				# sum of all neighbours: Between each candidate and population up to i
				for j in range(sample_size):
					neighbours_lengths[j] = sum(
						heuristic_candidates_sorted[tournament_players_indices[j]].neigh_edges(self.population[k]) for k in
						range(i)) / i

				max_index = np.argmax(neighbours_lengths)
				self.population[i] = deepcopy(heuristic_candidates_sorted[tournament_players_indices[max_index]])
				self.objective_values[i] = heuristic_candidates_obj_values_sorted[tournament_players_indices[max_index]]

		return self.population

	def parse(self, input, start, stop):
		if start > stop:
			return np.concatenate([input[start:], input[:stop]])
		return input[start:stop]

	# Crossover
	def order_recombination_wrap(self, parents):
		"""
		:param parents all parent=2+mu
		:return: all offspring with its objective values
		"""
		# st = time.time()
		offspring = np.empty(self.parameters.mu, dtype=Individual)
		offspring_objective_values = np.zeros(self.parameters.mu, dtype=np.double)
		for k in range(self.parameters.mu):
			i = random.randint(0, self.num_cities - 1)
			j = random.randint(0, self.num_cities - 1)
			p1, p2 = parents[2*k], parents[2*k+1]
			slice_P1 = self.parse(p1.order, i, j)
			candidate_P2 = np.roll(p2.order, -(j))
			no_dup = np.setdiff1d(candidate_P2, slice_P1, assume_unique=True)
			not_ordered_sol = np.concatenate([slice_P1, no_dup])
			sol = np.roll(not_ordered_sol, i)
			# Add self adaptivity
			if self.settings[5] == 1:
				# Update normal mutation rate
				beta = 2.0 * random.random() - 0.5  # map to interval [-0.5, 1.5]
				sol_alpha = max(0.05, p1.alpha + beta * (p2.alpha - p1.alpha))
				# Update flip rate
				sol_flip_rate = max(0.1, p1.flip_rate + beta * (p2.flip_rate - p1.flip_rate))
				offspring[k] = Individual(order=sol, alpha=sol_alpha, flip_rate=sol_flip_rate)
			else:
				offspring[k] = Individual(order=sol)
			# Store objective values of an offspring
			offspring_objective_values[k] = self.objf(offspring[k].order)
		# self.crossover_time += time.time() - st
		# print("order rec")
		# print(offspring_objective_values[offspring_objective_values < 0])
		return offspring, offspring_objective_values

	def scx_recombination(self, parents):
		# st = time.time()
		offspring = np.empty(self.parameters.mu, dtype=Individual)
		offspring_objective_values = np.zeros(self.parameters.mu, dtype=np.double)
		for k in range(self.parameters.mu):
			p1, p2 = parents[2*k], parents[2*k+1]
			child = np.zeros(self.num_cities, dtype=np.int32)
			child[0] = p1.order[0]
			child_obj_value = 0.0
			used_cities = {child[0]}
			p1_dict, p2_dict = p1.get_edges_as_dict(), p2.get_edges_as_dict()
			for i in range(1, self.num_cities):
				leg1 = p1_dict[child[i-1]]
				while leg1 in used_cities:
					leg1 = p1_dict[leg1]
				leg2 = p2_dict[child[i-1]]
				while leg2 in used_cities:
					leg2 = p2_dict[leg2]

				if self.weights[child[i-1], leg1] < self.weights[child[i-1], leg2]:
					child[i] = leg1
					child_obj_value += self.weights[child[i-1], leg1]
				else:
					child[i] = leg2
					child_obj_value += self.weights[child[i-1], leg2]
				used_cities.add(child[i])
			# Self adaptivity
			if self.settings[5] == 1:
				# Update normal mutation rate
				beta = 2.0 * random.random() - 0.5  # map to interval [-0.5, 1.5]
				sol_alpha = max(0.05, p1.alpha + beta * (p2.alpha - p1.alpha))
				# Update flip mutation rate
				sol_flip_rate = max(0.1, p1.flip_rate + beta * (p2.flip_rate - p1.flip_rate))
				offspring[k] = Individual(order=child, alpha=sol_alpha, flip_rate=sol_flip_rate)
			else:
				offspring[k] = Individual(order=child)
			offspring_objective_values[k] = child_obj_value + self.weights[child[self.num_cities-1], child[0]]
		# self.crossover_time += time.time() - st
		return offspring, offspring_objective_values

	# Mutation

	def flip_mutation(self, population, objective_values, current_best_index):
		"""
		Randomly choose 2 genes and swap their places.
		:param population: population of individuals
		:param objective_values:
		:param current_best_index:
		:return:
		"""
		random_mutation_rates = np.random.random(population.shape[0])
		for i, ind in enumerate(population):
			if random_mutation_rates[i] > ind.flip_rate or current_best_index == i:
				continue
			positions = np.random.choice(self.num_cities, replace=False, size=2)
			pos1, pos2 = min(positions), max(positions)
			prev, after = None, None  # before pos1 and after pos2
			if pos1 > 0:
				prev = pos1 - 1
			else:
				prev = self.num_cities - 1
			if pos2 < self.num_cities - 1:
				after = pos2 + 1
			else:
				after = 0

			old_fit, new_fit = 0.0, 0.0
			old_fit = self.weights[ind.order[prev], ind.order[pos1]] + self.weights[ind.order[pos1], ind.order[pos1+1]]
			if pos2 - pos1 > 1:
				old_fit += self.weights[ind.order[pos2-1], ind.order[pos2]]
				new_fit += self.weights[ind.order[pos2], ind.order[pos1+1]] + self.weights[ind.order[pos2-1], ind.order[pos1]]
			else:
				new_fit += self.weights[ind.order[pos2], ind.order[pos1]]

			if pos1 != 0 or pos2 != self.num_cities-1:
				old_fit += self.weights[ind.order[pos2], ind.order[after]]
				new_fit += self.weights[ind.order[pos1], ind.order[after]] + self.weights[ind.order[prev], ind.order[pos2]]
			else:
				new_fit += self.weights[ind.order[pos1], ind.order[pos2]]

			# Swap places
			ind.order[pos1], ind.order[pos2] = ind.order[pos2], ind.order[pos1]
			objective_values[i] = objective_values[i] - old_fit + new_fit

			# values_equal = abs(objective_values[i] - self.objf(population[i].order)) < 0.01
			# if not values_equal:
			# 	print("***ERROR LSO FLIP MUTATION***")

	def insert_mutation(self, population, objective_values, current_best_index):
		"""
		Population is here parent population + all offsprings. Performs insert mutation
		:param population:
		:param objective_values of a population
		:param current_best_index: index of currently best individual in population
		"""
		# Create 2D numpy array with 2 columns for two index
		random_mutation_rates = np.random.random(population.shape[0])
		# st = time.time()
		for i in range(population.shape[0]):
			if random_mutation_rates[i] > population[i].alpha or i == current_best_index:  # don't mutate vs mutate
				continue
			# Mutation logic
			positions = np.random.choice(self.num_cities, replace=False, size=2)
			pos1, pos2 = min(positions), max(positions)
			if pos2 - pos1 == 1:
				if pos2 < self.num_cities - 1:
					pos2 += 1
				else:
					pos1 -= 1
			betw = np.copy(population[i].order[pos1 + 1:pos2])
			betw = np.insert(betw, obj=0, values=population[i].order[pos2])
			j_aft = pos2 + 1
			if pos2 == self.num_cities - 1:
				j_aft = 0
			# Change obj value with condition
			objective_values[i] -= self.weights[population[i].order[pos2], population[i].order[j_aft]]
			objective_values[i] += self.weights[population[i].order[pos2-1], population[i].order[j_aft]]
			# Change objective value - this will always change
			objective_values[i] -= self.weights[population[i].order[pos1], population[i].order[pos1+1]]
			objective_values[i] -= self.weights[population[i].order[pos2-1], population[i].order[pos2]]
			# Introduce new weights
			objective_values[i] += self.weights[population[i].order[pos1], population[i].order[pos2]]
			objective_values[i] += self.weights[population[i].order[pos2], population[i].order[pos1+1]]
			# Change final order
			population[i].order[pos1 + 1:pos2 + 1] = betw
			# values_equal = abs(objective_values[i] - self.objf(population[i].order)) < 0.01
			# if not values_equal:
			# 	print("***ERROR LSO insert values***")
			# print("insert mutation")
			# print(objective_values[objective_values < 0])
		# self.insert_mutation_time += time.time() - st

	def scramble_mutation(self, population, objective_values, current_best_index):
		"""
		Scramble mutation implementation. Take a subset and randomly permutate part of the chromosome.
		:param population:
		:param current_best_index index of currently best individual in population
		:return:
		"""
		# st = time.time()
		# Positions
		arr = np.random.randint(low=0, high=self.num_cities, size=(population.shape[0], 2))
		random_mutation_rates = np.random.random(population.shape[0])
		for i in range(population.shape[0]):
			if random_mutation_rates[i] > population[i].alpha or i == current_best_index:  # don't mutate vs mutate
				continue
			# Swap min and max positions if necessary
			if arr[i, 0] > arr[i, 1]:
				tmp = arr[i, 0]
				arr[i, 0] = arr[i, 1]
				arr[i, 1] = tmp
			if arr[i, 1] - arr[i, 0] == 1:
				if arr[i, 1] < self.num_cities - 1:
					arr[i, 1] += 1
				else:
					arr[i, 0] -= 1
			elif arr[i, 1] - arr[i, 0] == 0:
				if arr[i, 1] < self.num_cities - 2:
					arr[i, 1] += 2
				else:
					arr[i, 0] -= 2
			betw = np.copy(population[i].order[arr[i, 0]:arr[i, 1]])
			np.random.shuffle(betw)
			# For appending edges and computing objective values
			neigh_bet = None  # neighbouring structure
			if arr[i, 0] > 0:  # handle if first element was selected
				neigh_bet = np.insert(betw, obj=0, values=population[i].order[arr[i, 0]-1])  # insert prior element
			else:
				neigh_bet = np.insert(betw, obj=0, values=population[i].order[self.num_cities-1])  # insert prior element
			neigh_bet = np.append(neigh_bet, population[i].order[arr[i, 1]])  # append from the index
			# betw_edges = population[i].compute_edges(neigh_bet)
			for j in range(arr[i, 1] - arr[i, 0] + 1):
				# population[i].edges[j] = betw_edges[cnt]
				objective_values[i] -= self.weights[population[i].order[j+arr[i, 0] - 1], population[i].order[j+arr[i, 0]]]
				objective_values[i] += self.weights[neigh_bet[j], neigh_bet[j+1]]
			# Change population order
			population[i].order[arr[i, 0]:arr[i, 1]] = betw
			# values_equal = abs(objective_values[i] - self.objf(population[i].order)) < 0.01
			# if not values_equal:
			# 	print("***ERROR LSO SCRAMBLE***")
			# print("scramble mutation")
			# print(objective_values[objective_values < 0])
		# self.scramble_mutation_time += time.time() - st

	def resample_population(self, population, objective_values, current_best_index):
		"""
		Leave only best few individuals and other change.
		:param population:
		:param objective_values: Objective values of given population
		:return:
		"""
		# st = time.time()
		N = 0.7
		for i in range(population.shape[0]):
			if i != current_best_index and random.random() < N:
				population[i] = Individual(num_cities=self.num_cities)
				objective_values[i] = self.objf(population[i].order)

		# self.rnd_mutation_time += time.time() - st
		return self.population, self.objective_values

	def lso_mutation(self, ind, obj_value):
		"""
		:param ind: ind is individual
		:return:
		"""
		x = np.random.randint(low=0, high=self.num_cities-3)
		old_fit = self.weights[ind.order[x], ind.order[x+1]] + self.weights[ind.order[x+1], ind.order[x+2]] + self.weights[ind.order[x+2], ind.order[x+3]]
		new_fit = self.weights[ind.order[x], ind.order[x+2]] + self.weights[ind.order[x+2], ind.order[x+1]] + self.weights[ind.order[x+1], ind.order[x+3]]
		if new_fit < old_fit:
			ind.order[x+1], ind.order[x+2] = ind.order[x+2], ind.order[x+1]
			return obj_value-old_fit+new_fit
		else:
			return obj_value

	def lso_mutation_pop(self, population, objective_values):
		for i, ind in enumerate(population):
			if random.random() < ind.alpha:
				objective_values[i] = self.lso_mutation(ind, objective_values[i])

	def increase_mutation_rate(self, population):
		"""
		You can do this differently
		:param population:
		:return:
		"""
		for ind in population:
			ind.alpha *= self.parameters.mutation_rate_coeff

	def lso2_opt_hard(self, population, objective_values, window_size_max):
		"""
		Objective values of given population. Performs
		:param population:
		:param objective_values:
		:param window_size_max:
		:return:
		"""
		for i in range(population.shape[0]):
			objective_values[i] = self.lso_2opt_ind(population[i], objective_values[i], window_size_max)
		# print("LSO 2 opt hard")
		# print(objective_values[objective_values < 0])
		return population, objective_values

	def lso2_opt_only_best(self, population, objective_values, window_size_max):
		N = 10
		sorted_indices = np.argsort(objective_values)[1:N]
		for index in sorted_indices:
			objective_values[index] = self.lso_2opt_ind(population[index], objective_values[index], window_size_max)
		return population, objective_values

	def lso2_opt_sample(self, population, objective_values, window_size_max):
		"""

		:param population:
		:param objective_values:
		:param window_size_max:
		:return:
		"""
		N = 10
		selected_indices = np.random.choice(population.shape[0], size=N, replace=True)  # with replacement - value can be selected multiple time
		for index in selected_indices:
			objective_values[index] = self.lso_2opt_ind(population[index], objective_values[index], window_size_max)

		# print("LSO 2 opt sample")
		# print(objective_values[objective_values < 0])
		return population, objective_values

	def lso_2opt_ind(self, ind, old_fit, window_size_max):
		"""
		Old fit is current fitness of an individual.
		:param ind:
		:param old_fit:
		:param window_size_max:
		:return: New fitness of an individual
		"""
		# Current fitness
		# st = time.time()
		roll_amount = np.random.randint(low=0, high=self.num_cities)
		ind.order = np.roll(ind.order, roll_amount)
		# ind.edges = np.roll(ind.edges, roll_amount)
		for i in range(self.num_cities-3):
			# I is the start of the interval
			a = i + self.parameters.window_size_start
			b = min(self.num_cities-1, i+self.parameters.window_size_start+window_size_max)
			improvements = 0
			for j in range(a, b):
				if improvements == 2:
					break
				curr_fit = self.weights[ind.order[j], ind.order[j+1]] + self.weights[ind.order[i], ind.order[i+1]]  # end edge
				new_fit = self.weights[ind.order[i+1], ind.order[j+1]] + self.weights[ind.order[i], ind.order[j]]

				# cached_edges = np.zeros(j-i-1)
				# st_loop_time = time.time()
				for k in range(i+1, j):
					curr_fit += self.weights[ind.order[k], ind.order[k+1]]
					new_fit += self.weights[ind.order[k+1], ind.order[k]]
					# cached_edges[k-i-1] = ind.order[k+1]*1000 + ind.order[k]
				# self.loop_time += time.time() - st_loop_time

				if curr_fit > new_fit > 0:
					# Change objective value
					old_fit = old_fit - curr_fit + new_fit
					ind.order[i + 1:j + 1] = np.flip(ind.order[i + 1:j + 1])
					improvements += 1

				# values_equal = abs(old_fit - self.objf(ind.order)) < 0.1
				# if not values_equal:
				# 	print("***ERROR LSO heur inside values***")

		ind.order = np.roll(ind.order, -roll_amount)
		# self.total_eval_time += time.time() - st
		return old_fit

	def lso_precise(self, individual, old_fit):
		"""
		Performs very precise local search for individual.
		:param individual:
		:param old_fit Old fitness of an individual
		:return:
		"""
		st = time.time()
		time_exceeded = False
		for i in range(self.num_cities-1):
			if time_exceeded is True:
				break
			for j in range(i+1, self.num_cities):
				# If new one is better keep it otherwise return it.
				# Calculate new fitness
				if j - i > 1:
					new_fit = old_fit - self.weights[individual.order[i], individual.order[i+1]] - self.weights[individual.order[j-1], individual.order[j]] \
						+ self.weights[individual.order[j], individual.order[i+1]] + self.weights[individual.order[j-1], individual.order[i]]
				else:
					new_fit = old_fit - self.weights[individual.order[i], individual.order[j]] + self.weights[individual.order[j], individual.order[i]]
				if i == 0 and j == self.num_cities - 1:
					new_fit -= self.weights[individual.order[self.num_cities-1], individual.order[0]]
					new_fit += self.weights[individual.order[0], individual.order[self.num_cities-1]]
				else:
					ind_bef = i - 1
					if i == 0:
						ind_bef = self.num_cities - 1
					new_fit -= self.weights[individual.order[ind_bef], individual.order[i]]
					new_fit += self.weights[individual.order[ind_bef], individual.order[j]]
					j_aft = j + 1
					if j == self.num_cities - 1:
						j_aft = 0
					new_fit -= self.weights[individual.order[j], individual.order[j_aft]]
					new_fit += self.weights[individual.order[i], individual.order[j_aft]]

				if old_fit > new_fit > 0:
					tmp = individual.order[i]
					individual.order[i] = individual.order[j]
					individual.order[j] = tmp
					old_fit = new_fit  # Modified fitness handling

				if time.time() - st > 2.0:
					time_exceeded = True
					print(f"Time passed: {time.time() - st}")
					break

					# values_equal = abs(old_fit - self.objf(individual.order)) < 0.01
		# self.lso_precise_time += time.time() - st
		return old_fit
	# Selection

	def several_lso_precise(self, population, objective_values):
		"""
		Apply lso precise on several individuals not just on best one.
		:param population:
		:param objective_values:
		:return:
		"""
		N = 5
		best_indices = np.argpartition(objective_values, range(N))[:N]
		for i in best_indices:
			objective_values[i] = self.lso_precise(population[i], objective_values[i])

	def several_lso_precise_sample(self, population, objective_values):
		"""
		Sample individuals on which it will be applied.
		:param population:
		:param objective_values:
		:return:
		"""
		N = 5
		indices = np.random.choice(population.shape[0], size=N, replace=False)
		for i in indices:
			objective_values[i] = self.lso_precise(population[i], objective_values[i])

	def exp_ranking_selection(self, curr_iter, objective_values, size):
		"""
		Exponential decay ranking. Returns 2*mu parents which will then later produce offspring
		:return: Indices from population that are going to be used.
		:param values values are not probabilities just exponential values
		:param objective_values: Objective values of given population
		"""
		s = pow(self.parameters.alpha, curr_iter)
		if s > 0:
			a = math.log(s) / (self.parameters.lambdaa - 1)
		else:
			a = 0
		values = self.lambda_indices * a
		values = np.exp(values)
		sm = sum(values)
		values = values / sm
		# print("Values: ", values)
		# indices = np.random.choice(self.lambda_indices, size=self.parameters.mu*2, p=values)
		indices = np.random.choice(self.lambda_indices, size=size, p=values)
		# print("Indices:", indices)
		# In indices are now stored selected indices

		# arg_sort_indices = np.argsort(self.objfpop(population))
		arg_sort_indices = np.argsort(objective_values)
		# return population[arg_sort_indices[indices]]
		return arg_sort_indices[indices]

	# The evolutionary algorithm's main loop
	def k_tournament_selection(self, objective_values, size):
		"""
		:param population: Population
		:param objective_values: objective values of given population
		:param size:
		:return:
		"""
		# selected_parents = np.empty(2*self.parameters.mu, dtype=Individual)
		selected_parents = np.empty(size, dtype=Individual)
		selected_indices = np.zeros(size)
		for ii in range(size):
			ri = random.choices(self.lambda_indices, k=self.parameters.k)  # saving k indexes from the population
			# min = np.argmin(self.objfpop(population[ri]))  # find best index
			min_ = np.argmin(objective_values[ri])
			# selected_parents[ii] = population[ri[min_]]
			selected_indices[ii] = ri[min_]
		return selected_parents

	def crowding_elimination(self, joined_population, objective_values):
		"""
		Crowding diversity promotion.
		:param joined_population:
		:param objective_values:
		:return:
		"""
		survivors = np.empty(self.parameters.lambdaa, dtype=Individual)
		survivors_obj_values = np.empty(self.parameters.lambdaa)
		perm = np.argsort(objective_values)
		indices = set(perm.flatten())
		i = 0
		cnt = 0
		N = int((self.parameters.lambdaa + self.parameters.mu) * 0.5)
		while cnt < self.parameters.lambdaa:
			# Individuals that will be used
			if perm[i] not in indices:
				i += 1
				continue
			# print(f"Test index: {perm[i]}")
			indices.remove(perm[i])  # remove because I put it already
			sampled_indices = np.random.choice(np.array(list(indices)), size=N, replace=False)  # without replacement - value cannot be selected multiple times
			ind, curr_min_dist = -1, 10000
			for j in range(N):
				dist = joined_population[perm[i]].neigh_edges(joined_population[sampled_indices[j]])
				if dist < curr_min_dist:
					ind, curr_min_dist = j, dist

			indices.remove(sampled_indices[ind])  # remove because he is the closest
			survivors[cnt] = joined_population[perm[i]]
			survivors_obj_values[cnt] = objective_values[perm[i]]
			# print(f"Selected on index: {perm[i]} Delete at: {sampled_indices[ind]}")
			cnt += 1
			i += 1
		return survivors, survivors_obj_values

	# Elimination
	def elimination(self, joined_population, objective_values):
		"""
		Performs elimination on joined population.
		:param joined_population:
		:param objective_values: All objective values from joined population.
		:return:
		"""
		# fvals = self.objfpop(joined_population)
		perm = np.argsort(objective_values)
		survivors = joined_population[perm[0:self.parameters.lambdaa]]
		survivors_objective_values = objective_values[perm[0:self.parameters.lambdaa]]
		return survivors, survivors_objective_values

	# Fitness sharing diversity promotion
	def fitness_sharing_obj(self, X, cached_distances, last_survivor, last_survivor_index):
		N = 20
		min_N_last = min(last_survivor_index, N)
		pop_indices = np.random.choice(last_survivor_index, size=min_N_last, replace=False)  # population indices
		x_N = self.parameters.lambdaa
		x_indices = np.random.choice(X.shape[0], size=x_N, replace=False)
		coeffs_sharing = np.zeros(x_N, dtype=np.double)
		coeffs_indices_X = np.zeros(x_N, dtype=np.uint32)
		cnt = 0
		for index in x_indices:
			x = X[index]
			one_plus_beta = 1.0  # initial value because each x is its own neighbourhood
			# st = time.time()
			for j in pop_indices:
				if cached_distances[index, j] <= self.parameters.sigma_sharing:
					one_plus_beta += 1 - (cached_distances[index, j] / self.parameters.sigma_sharing) ** self.parameters.alpha_sharing

			dist_last_survivor = x.neigh_edges(last_survivor)
			if dist_last_survivor <= self.parameters.sigma_sharing:
				one_plus_beta += 1 - (dist_last_survivor / self.parameters.sigma_sharing) ** self.parameters.alpha_sharing
			# self.elimination_time += time.time() - st
			cached_distances[index, last_survivor_index] = dist_last_survivor
			coeffs_sharing[cnt] = one_plus_beta  # it's always positive so no need for taking sign coefficient
			coeffs_indices_X[cnt] = index
			cnt += 1
		return coeffs_sharing, coeffs_indices_X

	def shared_elimination(self, population, objective_values):
		"""
		Fitness sharing elimination procedure.
		:param population:
		:param objective_values: Objective values of given population.
		:return:
		"""
		survivors = np.empty(self.parameters.lambdaa, dtype=Individual)
		survivors_obj_values = np.empty(self.parameters.lambdaa)
		cached_distances = np.empty(shape=(population.shape[0], self.parameters.lambdaa), dtype=np.uint32)
		used_indices = set()
		for i in range(self.parameters.lambdaa):
			if i < 1:
				index = np.argmin(objective_values)
				survivors[0] = population[index]  # take best
				survivors_obj_values[0] = objective_values[index]
				used_indices.add(index)
			else:
				coeffs_sharing, sampled_pop_indices = self.fitness_sharing_obj(population, cached_distances, survivors[i-1], i-1)
				# st = time.time()
				new_objective_values = np.multiply(np.take(objective_values, sampled_pop_indices), coeffs_sharing, dtype=np.double)
				N = 10
				temp_indices = np.argpartition(new_objective_values, N)
				j = 0
				while sampled_pop_indices[temp_indices[j]] in used_indices and j < N:
					j += 1
				if j < N:
					# print(f"NOT COPYING: {i} {temp_indices[j]}")
					survivors[i] = population[sampled_pop_indices[temp_indices[j]]]
					survivors_obj_values[i] = objective_values[sampled_pop_indices[temp_indices[j]]]
					used_indices.add(sampled_pop_indices[temp_indices[j]])
				else:
					# print(f"COPYING: {i} {temp_indices[0]}")
					survivors[i] = deepcopy(population[sampled_pop_indices[temp_indices[0]]])
					survivors_obj_values[i] = objective_values[sampled_pop_indices[temp_indices[0]]]
					used_indices.add(sampled_pop_indices[temp_indices[0]])
				# self.e_time2 += time.time() - st

		return survivors, survivors_obj_values

	def get_all_edges(self, pop):
		# st = time.time()
		for i in range(pop.shape[0]):
			pop[i].compute_all_edges()
		# print(f"Get all edges took: {time.time()-st:.2f}")

	def print_time_properties(self):
		print(f"Inverse mutation time: {self.inverse_mutation_time}")
		print(f"Scramble mutation time: {self.scramble_mutation_time}")
		print(f"Insert mutation time: {self.insert_mutation_time}")
		print(f"Random mutation time: {self.rnd_mutation_time}")
		print(f"Lso 2 opt time: {self.lso_2opt_time}")
		print(f"Lso precise time: {self.lso_precise_time}")
		print(f"Crossover time: {self.crossover_time}")
		print(f"Total eval time: {self.total_eval_time}")
		print(f"Loop total time: {self.loop_time}")
		print(f"Elimination time: {self.elimination_time}")
		print(f"E time2: {self.e_time2}")
		print(f"E time3: {self.e_time3}")
		print()

	def test_that_best_cycle_is_valid(self, current_best_value, best_candidate):
		if abs(self.objf(best_candidate.order) - current_best_value) > 1e-3:
			print("Invalid solution, debug needed!", current_best_value, self.obj(best_candidate))

	def optimize(self, filename):
		"""
		:param filename:
		:return:
		"""
		# Load your data
		# self.reporter.startTime = time.time()
		test_file = open(filename)
		self.weights = np.loadtxt(test_file, delimiter=",", dtype=np.double)
		self.weights[self.weights == np.inf] = 1e7  # comment that
		# Indices of ordered weights
		self.ordered_weights_indices = np.argsort(self.weights, axis=1)
		test_file.close()
		# Init
		self.num_cities = self.weights.shape[0]

		# st = time.time()
		# print(f"Population size: {self.parameters.lambdaa}")
		# self.population = self.heur_init()
		self.population = self.partial_population_init()
		# print("After initialization..")
		# print("Mean: ", np.mean(self.objective_values), "Best: ", np.min(self.objective_values))
		# print(f"Initialization took: {time.time()-st:.2f}")
		# st = time.time()
		self.population, self.objective_values = self.lso2_opt_hard(self.population, self.objective_values, self.parameters.window_size_max)
		# print(f"Local improvement took: {time.time()-st:.2f}")
		# print("After local improvement: ")
		# print("Mean: ", np.mean(self.objective_values), "Best: ", np.min(self.objective_values))
		# print(f"Time spent on evaluation: {self.total_eval_time}")

		# Average values per generations and best values per generation
		means = np.zeros(self.parameters.iterations)
		best_values = np.zeros(self.parameters.iterations)
		best_cycles = np.zeros((self.parameters.iterations, self.num_cities))

		# Index in population of best individual
		current_best_index = np.argmin(self.objective_values)

		# Old time left from reporter
		last_time_left = 300
		# Number of stagnating iterations
		stagnating = 0
		# Last time lso_precise was used
		last_checked = None
		# rnd_mode = 0  # random boost mode activated
		# elim_mode_dur = 0  # number of iterations - how long shared elimination will be applied

		stopping_criteria_times = 0  # if for 5 times mean and best result get to the same value we can stop

		for i in range(self.parameters.iterations):

			all_parents, offspring = None, None
			all_parents_indices = None  # indices of all parents that are going to be selected
			offspring_objective_values = None  # objective values of offspring
			if self.settings[1] == 0:
				all_parents_indices = self.k_tournament_selection(self.objective_values, 2*self.parameters.mu)
			elif self.settings[1] == 1:
				all_parents_indices = self.exp_ranking_selection(i, self.objective_values, 2*self.parameters.mu)

			all_parents = self.population[all_parents_indices]

			if last_time_left > 150:
				offspring, offspring_objective_values = self.order_recombination_wrap(all_parents)
			else:
				offspring, offspring_objective_values = self.scx_recombination(all_parents)

			# Apply LSO
			offspring, offspring_objective_values = self.lso2_opt_only_best(offspring, offspring_objective_values,
																					self.parameters.window_size_max)

			# DO NOT USE INVERSE MUTATION
			if i % 2 == 0:
				self.scramble_mutation(offspring, offspring_objective_values, None)
				self.scramble_mutation(self.population, self.objective_values, current_best_index)
			elif i % 2 == 1:
				self.insert_mutation(offspring, offspring_objective_values, None)
				self.insert_mutation(self.population, self.objective_values, current_best_index)
			else:
				self.flip_mutation(offspring, offspring_objective_values, None)
				self.flip_mutation(self.population, self.objective_values, current_best_index)

			# Join together population and objective values
			joined_population = np.concatenate((offspring, self.population), axis=0)
			joined_objective_values = np.concatenate((offspring_objective_values, self.objective_values), axis=0)

			self.population, self.objective_values = self.crowding_elimination(joined_population, joined_objective_values)

			# Current objective values and current best
			current_best_index = np.argmin(self.objective_values)
			best_cycles[i] = self.population[current_best_index].order

			# Because if they are the same we already checked it in last iteration.
			# We check in iteration i-1 so it can be a little bit cheaper than calculate mean two times
			# To not repeat yourself
			if self.lso_precise_iteration_time < last_time_left and\
					(last_time_left < 100 or (i > 0 and abs(means[i-1] - best_values[i-1]) < 50)):  # and if enough time
				print("Entering lso precise")
				st = time.time()
				if last_checked is None or np.array_equal(last_checked, best_cycles[i]) is False:
					last_checked = np.copy(self.population[current_best_index].order)
					self.several_lso_precise(self.population, self.objective_values)
				elif last_checked is None or np.array_equal(last_checked, best_cycles[i]) is True:
					last_checked = np.copy(self.population[current_best_index].order)
					self.several_lso_precise_sample(self.population, self.objective_values)

				best_cycles[i] = self.population[current_best_index].order

				self.lso_precise_iteration_time = time.time() - st

			mean_objective = np.mean(self.objective_values)
			means[i] = mean_objective
			best_values[i] = self.objective_values[current_best_index]

			timeLeft = self.reporter.report(mean_objective, self.objective_values[current_best_index],
											self.population[current_best_index].order)
			last_time_left = timeLeft

			# Stagnation criteria
			if i > 0 and np.array_equal(best_cycles[i-1], best_cycles[i]) is True:
				stagnating += 1
			else:
				# rnd_mode = 0
				stagnating = 0

			if abs(best_values[i] - means[i]) < 50 or stagnating == 5:
				# print("HEY")
				self.resample_population(self.population, self.objective_values, current_best_index)
				# rnd_mode = 1

			# self.test_that_best_cycle_is_valid(best_values[i], self.population[current_best_index])
			print(f"{i}. Mean objective: {mean_objective} Best fitness: {best_values[i]} Time left: {timeLeft}")

			if abs(means[i] - best_values[i]) < 1e-5:
				stopping_criteria_times += 1

			if stopping_criteria_times == 5 or timeLeft < 0:
				# self.print_time_properties()
				return means[0:i+1], best_values[0:i+1], best_cycles[0:i+1, :]

		# self.print_time_properties()
		# self.test_that_best_cycle_is_valid(best_values[current_best_index], self.population[current_best_index])
		return means, best_values, best_cycles
