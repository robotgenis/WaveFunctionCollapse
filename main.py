import multiprocessing
import time
import run, visual




# Define the functions to run
def func1(*args):
    print("Running function 1")
    run.run(*args)
    print("Finished running function 1")

def func2(*args):
    print("Running function 2")
    visual.run(*args)
    print("Finished running function 2")

if __name__ == '__main__':
	multiprocessing.freeze_support()

	m = multiprocessing.Manager()
	r = m.list()
	# Create processes for each function
	p1 = multiprocessing.Process(target=func1, args=(r, run.OUTPUT_X, run.OUTPUT_Y))
	p2 = multiprocessing.Process(target=func2, args=(r, run.OUTPUT_X, run.OUTPUT_Y))

	# Start the processes
	p1.start()
	p2.start()

	# Wait for the processes to finish
	p1.join()
	p2.join()

	# Done
	print("Done running functions")
