# Comments generated with https://chat.openai.com/chat

from random import choice, shuffle, choices
from copy import deepcopy
from collections import defaultdict
from queue import PriorityQueue
import sys

sys.setrecursionlimit(10000)

# Default Colors

COLORS = { # r.g.b.c.y.m.w
	"0": (255,   0,   0, 255),
	"1": (  0,   0, 255, 255),
	"2": (  0, 255,   0, 255),
	"3": (  0, 179, 179, 255),
	"4": (255,   0, 255, 255),
	"5": (255, 255,   0, 255),
	"6": (255, 255, 255, 255),
}
ROTATION = 1
MIRRORING_HORZ = 1
MIRRORING_VERT = 1



input_str = """
220000000
112222000
001111222
000000111
200000000
122200000
011120000
000012222
000001111
000000000
"""
N = 3
ROTATION = 0
MIRRORING_HORZ = 0
MIRRORING_VERT = 0



# input_str = """
# 1111
# 1000
# 1020
# 1000
# """
# N = 2
# ROTATION = 1
# MIRRORING_HORZ = 1
# MIRRORING_VERT = 1

# COLORS = {
# 	"0": (0, 0, 255, 255),
# 	"1": (111, 255, 55, 255),
# 	"2": (255, 255, 10, 255),
# 	"3": (111, 78, 55, 255),
# 	"4": (50, 30, 30, 255),
# 	"5": (255, 0, 0, 255),
# }
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




OUTPUT_X = 50
OUTPUT_Y = 50
# OUTPUT_X = 20
# OUTPUT_Y = 20
# OUTPUT_X = 5
# OUTPUT_Y = 5


class TileLocation:
	def __init__(self, N:int, x:int, y:int, tile_count:int):
		self.N = N
		self.x = x
		self.y = y
		self.tiles = [1 for _ in range(tile_count)]
		self.state = None
	
	def collapse(self, tiles:dict, tile_type_from_id:dict, global_tile_counts:list[int]):
		poss = [i for i in range(len(self.tiles)) if self.tiles[i]]
		weights = [tiles[tile_type_from_id[i]] / (global_tile_counts[i] + 1) for i in range(len(self.tiles)) if self.tiles[i]]

		self.state = choices(poss, weights)[0]

		self.tiles = [0 if i != self.state else 1 for i in range(len(self.tiles))]

	# Returns true if values changed
	def propagate(self, other, tile_type_from_id):
		dx = other.x - self.x
		dy = other.y - self.y

		poss1 = [0 for _ in range(len(self.tiles))]
		poss2 = [0 for _ in range(len(other.tiles))]

		for i1 in range(len(self.tiles)):
			if not self.tiles[i1]: continue

			t1 = tile_type_from_id[i1]

			for i2 in range(len(other.tiles)):
				if not other.tiles[i2]: continue

				t2 = tile_type_from_id[i2]

				overlapMatch = True

				# Check tile intersection
				for ox in range(N - abs(dx)):
					t1x = max(0, dx) + ox
					t2x = max(0, -dx) + ox
					for oy in range(N - abs(dy)):
						t1y = max(0, dy) + oy
						t2y = max(0, -dy) + oy
						if t1[t1y][t1x] != t2[t2y][t2x]:
							overlapMatch = False
					if not overlapMatch: break
				
				if overlapMatch:
					poss1[i1] = 1
					poss2[i2] = 1

		change1 = self.tiles != poss1
		change2 = other.tiles != poss2

		self.tiles = poss1
		other.tiles = poss2

		return change1, change2

	def __getitem__(self, k):
		return self.tiles[k]
	def __setitem__(self, k, v):
		self.tiles[k] = v
	def __contains__(self, k):
		return self.tiles[k]
	def __len__(self):
		return sum(self.tiles)
	def __iter__(self):
		return iter(i for i in range(len(self.tiles)) if self.tiles[i] == 1)

	def pointsContained(self):
		return [(dx, dy) for dx in range(self.x, self.x + self.N) for dy in range(self.y, self.y + self.N)]
	
	def __repr__(self):
		return f"T({self.x}, {self.y})-{sum(self.tiles)}" #-{self.state if self.state != None else 'X'}"

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

	block_types = set(i for row in input_map for i in row)
	block_count = len(block_types)
	block_type_from_id = {b: a for a, b in zip(list(block_types), range(block_count))}
	block_id_from_type = {block_type_from_id[a]: a for a in block_type_from_id}

	# print(block_type_from_id)
	# print(block_id_from_type, block_type_from_id, sep='\n')

	for x in range(input_width - N + 1):
		for y in range(input_height - N + 1):
			t = tuple(tuple(block_id_from_type[k] for k in i[x:x+N]) for i in input_map[y:y+N])
			for i in createRotations(t):
				tiles[i] += 1

	tile_count = len(tiles)
	tile_type_from_id = {b: a for a, b in zip(list(tiles), range(tile_count))}
	tile_id_from_type = {tile_type_from_id[a]: a for a in block_type_from_id}

	block_tile_map = {i: [k for k in tiles if any(i in row for row in k)] for i in range(block_count)}

	print(f"Generated {len(tiles)} Tiles")

	return tiles, tile_count, tile_type_from_id, tile_id_from_type, block_count, block_type_from_id, block_id_from_type, block_tile_map

def run(referenceGlobal, OX:int, OY:int):
	main(referenceGlobal, input_str, N, ROTATION, MIRRORING_HORZ, MIRRORING_VERT, OX, OY, COLORS)

# referenceGlobal, N, ROTATION, MIRRORING_HORZ, MIRRORING_VERT, OUTPUT_X, OUTPUT_Y
def main(referenceGlobal, IS:str, N:int, R:bool, MH:bool, MV:bool, OX:int, OY:int, c:dict[tuple[int,int,int,int]]):
	
	input_str, ROTATION, MIRRORING_HORZ, MIRRORING_VERT, OUTPUT_X, OUTPUT_Y, COLORS = IS, R, MH, MV, OX, OY, c
	
	tiles, tile_count, tile_type_from_id, tile_id_from_type, block_count, block_type_from_id, block_id_from_type, block_tile_map = createTiles(input_str, ROTATION, MIRRORING_HORZ, MIRRORING_VERT)

	wave = [[TileLocation(N, x, y, tile_count) for x in range(OUTPUT_X)] for y in range(OUTPUT_Y)]

	connectTiles = [[set() for _ in range(OUTPUT_X)] for _ in range(OUTPUT_Y)]

	for x in range(len(wave)):
		for y in range(len(wave[x])):
			for point in wave[y][x].pointsContained():
				if point[0] < OUTPUT_X and point[1] < OUTPUT_Y:
					connectTiles[point[1]][point[0]].add((x, y))

	def saveWave(wave):
		referenceGlobal[:] = [wave, N, COLORS, tile_type_from_id, block_type_from_id]

	def solve(wave:list[list[dict[tuple, TileLocation]]]):
		# Generate count of possibilities
		entropy = [[0 for _ in range(OUTPUT_X)] for _ in range(OUTPUT_Y)]
		
		minimumEntropy = float('inf')

		win = True
		for out_x in range(OUTPUT_X):
			for out_y in range(OUTPUT_Y):
				total = len(wave[out_y][out_x])	
					
				if total == 0: return # Lose condition (there are no possibilities)

				if total != 1: win = False # Win tracking
				
				entropy[out_y][out_x] = total	

				if total < minimumEntropy and total != 1:
					minimumEntropy = total

		if win: return wave # Win condition (everything only has one possibility)

		fillPoints = []
		
		for out_x in range(OUTPUT_X):
			for out_y in range(OUTPUT_Y):
				if entropy[out_y][out_x] == minimumEntropy:
					fillPoints.append((out_x,out_y))

		x, y = choice(fillPoints)

		#Calculate total occurance for weights
		placedTileCounts = [0 for _ in range(tile_count)]
		for out_x in range(OUTPUT_X):
			for out_y in range(OUTPUT_Y):
				if wave[y][x].state != None:
					placedTileCounts[wave[out_y][out_x].state] += 1
		wave[y][x].collapse(tiles, tile_type_from_id, placedTileCounts)

		# Propergate Loop
		propQueue = PriorityQueue()
		propQueue.put((1, x, y))
		# propQueue = [(x, y)]

		# while propQueue:
		while not propQueue.empty():
		
			_, curr_x, curr_y = propQueue.get_nowait()
			# curr_x, curr_y = propQueue.pop()

			tile1 = wave[curr_y][curr_x]

			for ox in range(-N + 1, N):
				t2x = curr_x + ox 

				if t2x < 0 or t2x >= OUTPUT_X: continue

				for oy in range(-N + 1, N):
					t2y = curr_y + oy

					if t2y < 0 or t2y >= OUTPUT_Y: continue

					tile2 = wave[t2y][t2x]
					change1, change2 = tile1.propagate(tile2, tile_type_from_id)

					if change1:
						propQueue.put((len(wave[curr_y][curr_x]), curr_x, curr_y))
						# propQueue.append((curr_x, curr_y))
					if change2:
						propQueue.put((len(wave[t2y][t2x]), t2x, t2y))
						# propQueue.append((t2x, t2y))


			saveWave(wave)

		return solve(wave)
	
	
	ans = solve(wave)
	saveWave(wave)
	print(wave)

if __name__ == "__main__":
	run([], OUTPUT_X, OUTPUT_Y)
	