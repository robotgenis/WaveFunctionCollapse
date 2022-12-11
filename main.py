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
ROTATION = 1
MIRRORING_HORZ = 1
MIRRORING_VERT = 1




COLORS = [
	(0,0,0,255),
	(255,255,255,255),
	(0,0,255,255)
]
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



# COLORS = [
# 	(0,0,0,255),
# 	(255,255,255,255),
# 	(0,255,0,255)
# ]
# input_str = """
# 220000000
# 112222000
# 001111222
# 000000111
# 200000000
# 122200000
# 011120000
# 000012222
# 000001111
# 000000000
# """
# N = 3
# ROTATION = 0
# MIRRORING_HORZ = 0
# MIRRORING_VERT = 0


# # Fails often on larger size
# COLORS = [
# 	(0,0,0,255),
# 	(255,255,255,255),
# 	(0,255,0,255)
# ]
# input_str = """
# 00000000000
# 22000000000
# 11222200020
# 00111122212
# 00000011101
# 20000000000
# 12220000000
# 01112000000
# 00001222200
# 00000111120
# 00000000120
# 00000000120
# """
# N = 3
# ROTATION = 1
# MIRRORING_HORZ = 1
# MIRRORING_VERT = 1




# COLORS = [
#     (0, 0, 255, 255),
# 	(111, 255, 55, 255),
# 	(255, 255, 10, 255),
# 	(111, 78, 55, 255),
# 	(50, 30, 30, 255),
# 	(255, 0, 0, 255)
# ]
# input_str = """
# 000000000000000000
# 000000000000000000
# 000000000000000000
# 000000000000000000
# 000000000000000000
# 000000000000000000
# 000000000000000000
# 000000111000000000
# 000001111100000000
# 000011111110000000
# 000011111110000000
# 000001111100000000
# 000000444000000000
# 000000444000000000
# 000000444000000000
# 333333333333333333
# 333333333333333333
# 333333333333333333
# 333333333333333333
# 333333333333333333
# 333333333333333333
# """
# N = 3
# ROTATION = 0
# MIRRORING_HORZ = 0
# MIRRORING_VERT = 0




# COLORS = [
# 	(0, 0, 255, 255),
# 	(111, 255, 55, 255),
# 	(255, 255, 10, 255),
# 	(111, 78, 55, 255),
# 	(50, 30, 30, 255),
# 	(255, 0, 0, 255),
# ]
# input_str = """
# 000000000000000000
# 000000000000000000
# 000000000000000000
# 000000000000000000
# 000000000000000000
# 000000000000000000
# 000000020000000000
# 000000010000000000
# 000000115000000000
# 000001111100000000
# 000011511110000000
# 000111111111000000
# 000000444000000000
# 000000444000000000
# 000000444000000000
# 333333333333333333
# 333333333333333333
# 333333333333333333
# 333333333333333333
# 333333333333333333
# 333333333333333333
# """
# N = 3
# ROTATION = 0
# MIRRORING_HORZ = 0
# MIRRORING_VERT = 0





OUTPUT_X = 150
OUTPUT_Y = 150

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
