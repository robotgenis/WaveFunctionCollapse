from random import choice, shuffle, choices
from copy import deepcopy
from queue import PriorityQueue

import sys
from collections import defaultdict
import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

sys.setrecursionlimit(10000)

COLORS = [ # r.g.b.c.y.m.w
	(255,   0,   0, 255),
	(  0,   0, 255, 255),
	(  0, 255,   0, 255),
	(  0, 179, 179, 255),
	(255,   0, 255, 255),
	(255, 255,   0, 255),
	(  0,   0,   0, 255),
	(255, 255, 255, 255)
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

OUTPUT_X = 256
OUTPUT_Y = 256



# # Create a CUDA kernel
# compute_entropy_module = SourceModule("""
# __global__ void compute_entropy(bool *wave, unsigned int *OUTPUT_X, unsigned int *OUTPUT_Y, unsigned int *tile_count, unsigned int *entropy) {
# 	const int x = threadIdx.x + blockIdx.x * blockDim.x;
# 	const int y = threadIdx.y + blockIdx.y * blockDim.y;
# 	const int chunk = x + y * blockDim.x * gridDim.x;

# 	int t, score = 0;
# 	for(t = 0; t < *tile_count; ++t){
# 		score += [chunk * tile_count + t];
# 	}

# 	entropy[chunk] = score;
# }
# """)
# compute_entropy = compute_entropy_module.get_function("compute_entropy")


def run(referenceGlobal, OX:int, OY:int):
	main(referenceGlobal, input_str, N, ROTATION, MIRRORING_HORZ, MIRRORING_VERT, OX, OY, COLORS)

# referenceGlobal, N, ROTATION, MIRRORING_HORZ, MIRRORING_VERT, OUTPUT_X, OUTPUT_Y
def main(referenceGlobal, IS_input:str, N_input:int, R:bool, MH:bool, MV:bool, OX_input:int, OY_input:int, c_input:list[tuple[int,int,int,int]]):

	colors_array = np.array(c_input, dtype=np.uint8)
	N = np.uint8(N_input)
	OUTPUT_X = np.uint32(OX_input)
	OUTPUT_Y = np.uint32(OY_input)
		
	def createTiles(input_str, ROTATION, MIRRORING_HORZ, MIRRORING_VERT):
		def inputStrToList(data:list[str]) -> list[list[int]]:
			# Convert the input string data to a list of lists of integers
			return [list(i) for i in data.strip().split("\n")]
		
		def rotate90Clockwise(A):
			# Get the length of one side of the square
			N = len(A[0])
			# Loop through the top half of the square
			for i in range(N // 2):
				for j in range(i, N - i - 1):
					# Store the element at the top left corner of the sub-square
					temp = A[i][j]
					# Swap the elements at the four corners of the sub-square
					A[i][j] = A[N - 1 - j][i]
					A[N - 1 - j][i] = A[N - 1 - i][N - 1 - j]
					A[N - 1 - i][N - 1 - j] = A[j][N - 1 - i]
					A[j][N - 1 - i] = temp
			# The square A has been rotated clockwise by 90 degrees
			return A

		def mirrorHorz(A):
			# Calculate the length of the first row of the input array
			l = len(A[0])
			# Loop through half of the array, from 0 to (l // 2) - 1
			for x in range(l // 2):
				# Loop through all the rows in the array
				for y in range(len(A)):
					# Swap the elements at the x and l-x-1 indices of the y-th row of A
					A[y][x], A[y][l-x-1] = A[y][l-x-1], A[y][x]

		def mirrorVert(A):
			# Calculate the length of the input array
			l = len(A)
			# Loop through half of the array, from 0 to (l // 2) - 1
			for y in range(l // 2):
				# Loop through all the columns in the array
				for x in range(len(A[0])):
					# Swap the elements at the y and l-y-1 indices of the x-th column of A
					A[y][x], A[l-y-1][x] = A[l-y-1][x], A[y][x]

		# tile must be square
		def createRotations(tile:tuple[tuple[int]]) -> list[tuple[tuple[int]]]:
			output = []
			arr = list(list(i) for i in tile)
			for _ in range(4):
				output.append(tuple(tuple(i) for i in arr))
				if MIRRORING_HORZ:
					mirrorHorz(arr)
					output.append(tuple(tuple(i) for i in arr))
				if MIRRORING_VERT:
					mirrorVert(arr)
					output.append(tuple(tuple(i) for i in arr))
				if MIRRORING_HORZ:
					mirrorHorz(arr)
					output.append(tuple(tuple(i) for i in arr))
				if MIRRORING_VERT:
					mirrorVert(arr)

				if not ROTATION: break
				rotate90Clockwise(arr)

			return output
		
		input_map = inputStrToList(input_str)
		input_width = len(input_map[0])
		input_height = len(input_map)

		tiles = defaultdict(lambda: 0)

		for x in range(input_width - N + 1):
			for y in range(input_height - N + 1):
				t = tuple(tuple(k for k in i[x:x+N]) for i in input_map[y:y+N])
				for i in createRotations(t):
					# TODO reduce count for rotation and mirroring
					tiles[i] += 1
		
		temp_tiles_array = []
		temp_tiles_array_counts = []
		for k,v in tiles.items():
			temp_tiles_array.append(k)
			temp_tiles_array_counts.append(v)
		
		tile_array = np.array(temp_tiles_array, np.uint8)
		tile_array_counts = np.array(temp_tiles_array_counts, dtype=np.uint32)
		
		# print(tiles_array)
		# print(tiles_array_counts)

		print(f"Generated {len(tile_array)} Tiles")

		return tile_array, tile_array_counts

	tile_array, tile_array_counts = createTiles(IS_input, R, MH, MV)

	tile_count = np.uint32(len(tile_array))

	wave = np.ones((OUTPUT_Y, OUTPUT_X, tile_count), dtype=bool)
	wave_output= np.ones((OUTPUT_Y, OUTPUT_X, tile_count), dtype=bool)

	entropy_array = np.ones((OUTPUT_Y, OUTPUT_X), dtype=np.uint32)

	# GPU Memory Definitions
	wave_gpu = drv.mem_alloc(wave.nbytes)
	drv.memcpy_htod(wave_gpu, wave)

	entropy_array_gpu = drv.mem_alloc(entropy_array.nbytes)
	drv.memcpy_htod(entropy_array_gpu, entropy_array)

	N_gpu = drv.mem_alloc(N.nbytes)
	drv.memcpy_htod(N_gpu, N)

	tile_count_gpu = drv.mem_alloc(tile_count.nbytes) # Final
	drv.memcpy_htod(tile_count_gpu, tile_count)

	tile_array_gpu = drv.mem_alloc(tile_array.nbytes) # Final
	drv.memcpy_htod(tile_array_gpu, tile_array)

	referenceGlobal[:] = [wave_output, N, tile_count, tile_array, colors_array]

	def saveWave():
		drv.memcpy_dtoh(wave_output, wave_gpu)

	saveWave()

	# def solve():
	# 	# Generate a list of possibility counts
		
		
		

if __name__ == "__main__":
	run([], OUTPUT_X, OUTPUT_Y)