from math import ceil

MAX_THREAD = [1024, 1024, 64]

MAX_BLOCK = (32, 32, 1)
WARP_SIZE = 32

def createsBlockGridSizes(*a):
	assert len(a) == 3
	block = [min(MAX_BLOCK[i], int(a[i])) for i in range(3)]
	grid = [ceil(a[i] / block[i]) for i in range(3)]
	return tuple(block), tuple(grid)


