from reporter import Reporter
import numpy as np


# Modify the class name to match your student number. Evolutionary algorithm
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
		self.population = None

	def initialize(self):
		"""
		Return population. Now it's very simple random initialization.
		:return:
		"""
		return np.random.randint(low=1, high=self.weights.shape[0] + 1, size=(self.lambdaa, self.weights.shape[0]))

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
		Population is here parent population + all offsprings.
		:param population:
		:return: mutated population
		"""
		pass

	# The evolutionary algorithm's main loop
	def selection(self):
		pass

	def optimize(self, filename):
		test_file = open(filename)
		self.weights = np.loadtxt(test_file, delimiter=",")
		test_file.close()
		self.population = self.initialize()  # Initialize population

		for i in range(self.iters):
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
	lambdaa = 100  # population size, maybe too much
	mu = 50  # also maybe too much we need to check
	iters = 100  # number of iterations to be run

	#weights = np.loadtxt(open(file_name), delimiter=",")

	#arr = np.random.randint(low=1, high=weights.shape[0] + 1, size=(lambdaa, weights.shape[0]))

	algorithm = r0123456(lambdaa, mu, alpha, iters)
	algorithm.optimize(file_name)









