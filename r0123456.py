from reporter import Reporter
import numpy as np
import random


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
		for i in range(self.lambdaa):
			self.population[i, :] = np.random.choice(np.arange(0, self.num_cities, dtype=np.uint32), replace=False, size=self.num_cities)
		return self.population

	def crossover(self, selected_population):
		# Performs crossover on selected population and returns only offspring
		return selected_population

	def mutation(self, population):
		"""
		Population is here parent population + all offsprings. Performs insert mutation
		:param population:
		:return: mutated population
		"""
		# Create 2D numpy array with 2 columns for two index
		for i in range(population.shape[0]):
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
		selected = np.zeros((self.mu, self.num_cities), dtype=np.uint32)
		for ii in range(self.mu):
			ri = random.choices(range(self.lambdaa), k = self.k)  # saving k indexes from the population
			min = np.argmin(self.objfpop(self.population[ri, :]))  # find best index
			selected[ii, :] = self.population[ri[min], :]
			# print("Selected weight:", self.objf(self.population[ri[min], :]))
		return selected

	def elimination(self, joined_population):
		"""
		Performs elimination on joined population.
		:param joined_population:
		:return:
		"""
		# print("Shape of joined population which came into elimination process:", joined_population.shape[0], joined_population.shape[1])
		fvals = self.objfpop(joined_population)
		# print("Fvals in elimination:", fvals)
		perm = np.argsort(fvals)
		# print("Perm in elimination:", perm)
		survivors = joined_population[perm[0:self.lambdaa], :]
		# print("Survivors shape:", survivors.shape[0], survivors.shape[1])
		return survivors

	def optimize(self, filename):
		test_file = open(filename)
		self.weights = np.loadtxt(test_file, delimiter=",")
		test_file.close()
		self.num_cities = self.weights.shape[0]
		self.population = self.initialize()  # Initialize population


		for i in range(self.iters):
			# print(f"Starting population in iteration {i+1}:", self.population)
			# print(f"Population shape at the beginning of iteration {i + 1}:", self.population.shape[0], self.population.shape[1])
			selected = self.selection()
			# print("Selected individuals:", selected)
			offspring = self.crossover(selected)
			# print("Offspring:", offspring)
			joined_population = np.vstack((self.mutation(offspring), self.population))
			# print("Joined population:", joined_population)
			# print("Joined population array shape:", joined_population.shape[0], joined_population.shape[1])
			self.population = self.elimination(joined_population)
			# print(f"Population shape at the end of iteration {i+1}:", self.population.shape[0], self.population.shape[1])

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

		# Your code here.
		return 0


if __name__ == "__main__":
	file_name = "./test/tour29.csv"
	alpha = 0.05  # mutation rate
	lambdaa = 50  # population size, maybe too much
	mu = 50  # also maybe too much we need to check
	iters = 100  # number of iterations to be run
	k = 5
	algorithm = r0123456(lambdaa=lambdaa, mu=mu, alpha=alpha, iters=iters, k=k)

	algorithm.optimize("./test/tour29.csv")


	# Mutation test
	# algorithm.population = algorithm.initialize()
	# print(algorithm.population)
	# algorithm.mutation(population=algorithm.population)










