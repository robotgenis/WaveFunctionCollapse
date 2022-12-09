from math import ceil

MAX_THREAD = [1024, 1024, 64]
WARP_SIZE = 32

def createsBlockGridSizes(*a):
	assert len(a) == 3
	block = [v for v in a]
	grid = [1 for _ in a]
	for i in range(len(block)):
		if block[i] > WARP_SIZE:
			block[i] = WARP_SIZE
			grid[i] = ceil(a[i] / WARP_SIZE)
	return tuple(block), tuple(grid)


