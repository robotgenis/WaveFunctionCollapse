import multiprocessing
import time
import generate, visual

# Define the functions to run
def func1(*args):
    print("Generating Pattern")
    generate.run(*args)
    print("Finished Generating Pattern")

def func2(*args):
    print("Running Visualizer")
    visual.run(*args)
    print("Closed Visualizer")

if __name__ == '__main__':
	OUTPUT_X, OUTPUT_Y = generate.OUTPUT_X, generate.OUTPUT_Y

	multiprocessing.freeze_support()

	m = multiprocessing.Manager()
	r = m.list()
	# Create processes for each function
	p1 = multiprocessing.Process(target=func1, args=(r, OUTPUT_X, OUTPUT_Y))
	p2 = multiprocessing.Process(target=func2, args=(r, OUTPUT_X, OUTPUT_Y))

	# Start the processes
	p1.start()
	p2.start()

	# Wait for the processes to finish
	p1.join()
	p2.join()

	# Done
	print("Done running functions")
