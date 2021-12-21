from reporter import Reporter
import numpy as np
import random
import os, psutil
import matplotlib.pyplot as plt
import time

# Modify the class name to match your student number. Evolutionary algorithm
# Git kraken
class r0123456:

	#def optimize(self, filename):
	# In constructor you should get mutation rate, number of iterations, population size, offspring size
	def __init__(self, lambdaa, mu, alpha, iters, k):
		# This is to be convenient with their code
		self.reporter = Reporter(self.__class__.__name__)
		self.lambdaa = lambdaa  # population size
		self.mu = mu  # offspring size
		self.alpha = alpha  # mutation raze
		self.iters = iters # number of iterations
		self.weights = None  # will be initialized later
		self.num_cities = None  # will be intialized later
		self.population = None
		self.k = k  # k-tournament parameter

	def objf(self, candidate):
		sum = 0
		for i in range(self.num_cities - 1):
			city_2 = candidate[i+1]
			city_1 = candidate[i]
			# print("City_2:", city_2, "city_1:", city_1)
			weight = self.weights[city_1, city_2]
			# print("Weight:", weight)
			sum += weight
		# print("Weight without first and last:", sum)
		last_and_first_weight = self.weights[candidate[self.num_cities - 1], candidate[0]]
		sum += last_and_first_weight
		# print("Final sum:", sum)
		return sum

	def objfpop(self, candidates):
		array = np.zeros(candidates.shape[0])
		for i in range(candidates.shape[0]):
			array[i] = self.objf(candidates[i])
		return array

	def initialize(self):
		"""
		Return population. Now it's very simple random initialization.
		:return:
		"""
		self.population = np.zeros((self.lambdaa, self.num_cities), dtype=np.uint32)
		# i = 0
		# while i < self.lambdaa:
		# 	rnd = np.random.choice(np.arange(0, self.num_cities, dtype=np.uint32), replace=False, size=self.num_cities)
		# 	if self.objf(rnd) != np.inf:
		# 		self.population[i, :] = rnd
		# 		i += 1
		for i in range(self.lambdaa):
			self.population[i, :] = np.random.choice(np.arange(0, self.num_cities, dtype=np.uint32), replace=False, size=self.num_cities)
		return self.population

	def parse(self, input, start, stop):
		if start > stop:
			return np.concatenate([input[start:], input[:stop]])
		return input[start:stop]

	def recombination(self, p1, p2):
		i = random.randint(0, self.num_cities - 1)
		j = random.randint(0, self.num_cities - 1)

		slice_P1 = self.parse(p1, i, j)
		candidate_P2 = np.roll(p2, -(j))
		no_dup = np.setdiff1d(candidate_P2, slice_P1, assume_unique=True)
		not_ordered_sol = np.concatenate([slice_P1, no_dup])
		sol = np.roll(not_ordered_sol, i)
		return sol

	def mutation(self, population):
		"""
		Population is here parent population + all offsprings. Performs insert mutation
		:param population:
		:return: mutated population
		"""
		# Create 2D numpy array with 2 columns for two index
		ii = np.where(np.random.rand(np.size(population, 0)) <= self.alpha)[0]
		for i in ii:
			# print(f"Population[{i}]", population[i, :])
			positions = np.random.choice(np.arange(0, population.shape[1], dtype=np.uint32), replace=False, size=2)
			pos1, pos2 = min(positions), max(positions)
			# print("Positions: ", (pos1, pos2))
			bef = population[i, 0:pos1 + 1]
			# print("Array before:", bef)
			bef = np.append(bef, population[i, pos2])
			# print("Array before:", bef)
			after = population[i, pos2 + 1:]
			# print("Array after:", after)
			between = population[i, pos1 + 1:pos2]
			# print("Array between:", between)
			solution = np.concatenate((bef, between, after))
			population[i] = solution
			# print("Solution:", solution)
			# print()
			# print()
		return population

	# The evolutionary algorithm's main loop
	def selection(self):
		ri = random.choices(range(self.lambdaa), k=self.k)  # saving k indexes from the population
		min = np.argmin(self.objfpop(self.population[ri, :]))  # find best index
		return self.population[ri[min], :]

	def elimination(self, joined_population):
		"""
		Performs elimination on joined population.
		:param joined_population:
		:return:
		"""
		fvals = self.objfpop(joined_population)
		perm = np.argsort(fvals)
		survivors = joined_population[perm[0:self.lambdaa], :]
		return survivors

	def optimize(self, filename):
		"""
		:param filename:
		:return:
		"""

		test_file = open(filename)
		self.weights = np.loadtxt(test_file, delimiter=",")
		self.weights[self.weights == np.inf] = 1e7  # comment that
		test_file.close()
		self.num_cities = self.weights.shape[0]
		self.population = self.initialize()  # Initialize population

		best_value, best_cycle = None, None

		mean_ar = []
		best_ar = []
		time_arr=[]
		old_time = time.time()
		for i in range(self.iters):
			offspring = np.zeros((self.mu, self.num_cities), dtype=np.uint32)
			for ii in range(self.mu):
				p1 = self.selection()
				p2 = self.selection()
				offspring[ii] = self.recombination(p1, p2)

			joined_population = np.vstack((self.mutation(offspring), self.mutation(self.population)))

			self.population = self.elimination(joined_population)

			# Your code here.
			# Call the reporter with:
			#  - the mean objective function value of the population
			#  - the best objective function value of the population
			#  - a 1D numpy array in the cycle notation containing the best solution
			#    with city numbering starting from 0
			objective_values = self.objfpop(self.population)
			mean_objective = np.mean(objective_values)
			best_objective_index = np.argmin(objective_values)

			timeLeft = self.reporter.report(mean_objective, objective_values[best_objective_index], self.population[best_objective_index])
			if timeLeft < 0:
				break

			# print('mean: ' + str(mean_objective) + '     best: ' + str(objective_values[best_objective_index]))

			if best_value is None or objective_values[best_objective_index] < best_value:
				best_value = objective_values[best_objective_index]
				best_cycle = self.population[best_objective_index]
			print('i'+ str(i) + '  mean: ' + str(mean_objective) + '           best: ' + str(objective_values[best_objective_index]))
			mean_ar.append(mean_objective)
			best_ar.append(objective_values[best_objective_index])
			new_time = time.time()
			time_arr.append(new_time-old_time)

		plot1 = plt.figure(1)
		plt.plot(mean_ar, label = "mean")
		plt.plot(best_ar, label = "best")
		plt.xlabel("iterations", fontsize=20)
		plt.ylabel("objective value", fontsize=20)
		plt.legend(prop={'size': 20})

		plot2 = plt.figure(2)
		plt.plot(time_arr, mean_ar, label="mean")
		plt.plot(time_arr, best_ar, label="best")
		plt.xlabel("time (s)", fontsize=20)
		plt.ylabel("objective value", fontsize=20)
		plt.legend(prop={'size': 20})

		plt.show()


		#print("Best value:", best_value)
		#print("Best cycle:", best_cycle)
		return 0

if __name__ == "__main__":
	filename = "./test/tour29.csv"
	# create TSP problem
	# hier kan een class gemaakt worden voor een random TSP problem met input file name
	# en num cities en weights enzo die hier terug te vinden zijn --> kijk naar het voorbeeld in de les van code
	# create parameters
	alpha = 0.07  # mutation rate
	lambdaa = 500  # population size, maybe too much
	mu = 500  # also maybe too much we need to check (TODO is dit offspring?)
	iters = 100  # number of iterations to be run
	k = 3 # for k-tournament selection
	algorithm = r0123456(lambdaa=lambdaa, mu=mu, alpha=alpha, iters=iters, k=k)
	algorithm.optimize(filename)

	# Print memory consumption in bytes
	process = psutil.Process(os.getpid())
	bytes_mem_cons = process.memory_info().rss # in bytes
	print(f"Memory in consumption in MB: {bytes_mem_cons / (1024*1024):.2f}")












