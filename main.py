import multiprocessing
import time
import generate, visual

# COLORS = [ # r.g.b.c.y.m.w
# 	(255,   0,   0, 255),
# 	(  0,   0, 255, 255),
# 	(  0, 255,   0, 255),
# 	(  0, 179, 179, 255),
# 	(255,   0, 255, 255),
# 	(255, 255,   0, 255),
# 	(  0,   0,   0, 255),
# 	(255, 255, 255, 255)
# ]

COLORS = [
	(0,0,0,255),
	(255,255,255,255),
	(0,0,255,255)
]

ROTATION = 1
MIRRORING_HORZ = 1
MIRRORING_VERT = 1

input_str = """
1111
1000
1020
1000
"""
N = 2
ROTATION = 1
MIRRORING_HORZ = 1
MIRRORING_VERT = 1

OUTPUT_X = 40
OUTPUT_Y = 40

# Define the functions to run
def func1(*args):
    print("Generating Pattern")
    generate.gen(*args)
    print("Finished Generating Pattern")

def func2(*args):
    print("Running Visualizer")
    visual.run(*args)
    print("Closed Visualizer")

if __name__ == '__main__':
	multiprocessing.freeze_support()

	m = multiprocessing.Manager()
	r = m.list()
	# Create processes for each function
	p1 = multiprocessing.Process(target=func1, args=(r, input_str, N, ROTATION, MIRRORING_HORZ, MIRRORING_VERT, OUTPUT_X, OUTPUT_Y, COLORS))
	p2 = multiprocessing.Process(target=func2, args=(r, OUTPUT_X, OUTPUT_Y))

	# Start the processes
	p1.start()
	p2.start()

	# Wait for the processes to finish
	p1.join()
	p2.join()

	# Done
	print("Done running functions")
