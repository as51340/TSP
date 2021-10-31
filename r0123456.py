from reporter import Reporter
import numpy as np


# Modify the class name to match your student number. Evolutionary algorithm
# Git kraken
class r0123456:

	#def optimize(self, filename):
	# In constructor you should get mutation rate, number of iterations, population size, offspring size
	def __init__(self, lambdaa, mu, alpha, iters):
		# This is to be convenient with their code
		self.reporter = Reporter(self.__class__.__name__)
		self.lambdaa = lambdaa  # population size
		self.mu = mu  # offspring size
		self.alpha = alpha  # mutation raze
		self.iters = iters # number of iterations
		self.weights = None  # will be initialized later
		self.num_cities = None  # will be intialized later
		self.population = None  # will be initialized later

	def initialize(self):
		"""
		Return population. Now it's very simple random initialization.
		:return:
		"""
		self.population = np.zeros((self.lambdaa, self.num_cities))
		for i in range(self.lambdaa):
			self.population[i, :] = np.random.choice(np.arange(1, self.num_cities + 1), replace=False, size=self.num_cities)
		return self.population

	def selection(self):
		"""
		Return one parent. Perform k-tournament selection
		:return:
		"""
		pass

	def crossover(self, individual1, individual2):
		"""
		Return two offsprings
		:param individual1:
		:param individual2:
		:return: two offsprings
		"""
		pass

	def mutation(self, population):
		"""
		Population is here parent population + all offsprings. Performs insert mutation
		:param population:
		:return: mutated population
		"""
		# Create 2D numpy array with 2 columns for two index
		for i in range(population.shape[0]):
			print(f"Population[{i}]", population[i, :])
			positions = np.random.choice(np.arange(0, population.shape[1]), replace=False, size=2)
			pos1, pos2 = max(positions), min(positions)
			print("Positions: ", (pos1, pos2))
			bef = population[i, 0:pos1 + 1]
			print("Array before:", bef)
			bef = np.append(bef, population[i, pos2])
			print("Array before:", bef)
			after = population[i, pos2 + 1]
			print("Array after:", after)
			between = population[i, pos1 + 1:pos2]
			print("Array between:", between)
			solution = np.concatenate((bef, between, after))
			print("Solution:", solution)

	def optimize(self, filename):
		test_file = open(filename)
		self.weights = np.loadtxt(test_file, delimiter=",")
		test_file.close()
		self.population = self.initialize()  # Initialize population
		self.num_cities = self.weights.shape[0]

		for i in range(self.iters):

			selected = self.selection()
			
			# Your code here.
			# Call the reporter with:
			#  - the mean objective function value of the population
			#  - the best objective function value of the population
			#  - a 1D numpy array in the cycle notation containing the best solution
			#    with city numbering starting from 0
			mean_objective, best_objective, best_solution = 0.0, 0.0, np.random.randint(low=1, high=11, size=10)

			timeLeft = self.reporter.report(mean_objective, best_objective, best_solution)
			if timeLeft < 0:
				break

		# Your code here.
		return 0


if __name__ == "__main__":
	file_name = "./test/tour29.csv"
	alpha = 0.05  # mutation rate
	lambdaa = 10  # population size, maybe too much
	mu = 50  # also maybe too much we need to check
	iters = 100  # number of iterations to be run
	algorithm = r0123456(lambdaa, mu, alpha, iters)
	algorithm.population = algorithm.initialize()
	print(algorithm.population)
	#algorithm.mutation(population)



	# algorithm.optimize(file_name)









