# Comments generated with https://chat.openai.com/chat

from random import choice, shuffle, choices
from copy import deepcopy
import sys

sys.setrecursionlimit(10000)

# Defualt Colors

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



# input_str = """
# 00100100100
# 00100100100
# 00112221111
# 00002220000
# 11112221100
# 00100010100
# 00100011311
# 00100000100
# 00100000100
# 00100000100
# 00100000100
# """
# N = 3
# ROTATION = 1
# MIRRORING_HORZ = 1
# MIRRORING_VERT = 1

input_str = """
112222000
001111222
000000111
200000000
122200000
011120000
000012222
000001111
"""
N = 3
ROTATION = 0
MIRRORING_HORZ = 0
MIRRORING_VERT = 0

# input_str = """
# 03000000030
# 01300100310
# 22232223222
# 01003130010
# 01000300010
# 01003130010
# 22232223222
# 01300100310
# 03000100030
# 31000100013
# """
# N = 3


# input_str = """
# 0000000000
# 0000000000
# 0001110000
# 0011111000
# 0022222000
# 0022222000
# 3333333333
# 0000000000
# 0000000000
# """
# N = 3



# input_str = """
# 1111
# 1000
# 1020
# 1020
# 0000
# """
# N = 2

# input_str = """
# 00000
# 01020
# 01020
# 00000
# """
# N = 2

# input_str = """
# 000001010
# 000010001
# 000101000
# 001000101
# 010000010
# 100000101
# """
# N = 3

# input_str = """
# 020001010
# 002010001
# 000201000
# 001020101
# 210002010
# 120020101
# """
# N = 3


# input_str = """
# 000000
# 010000
# 000200
# 030000
# 000000
# """
# N = 2




# input_str = """
# 0000000000
# 0101020202
# 0000000000
# 0101020202
# 0000000000
# 0102020303
# 0000000000
# 0102030303
# """
# N = 3


# input_str = """
# 000000000000
# 000000000000
# 011100000110
# 011111001110
# 011111111100
# 011111111000
# 001111110000
# 000001100000
# """
# N = 3

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
# N = 4
# ROTATION = 0
# MIRRORING_HORZ = 0
# MIRRORING_VERT = 0



# COLORS = {
# 	"0": (0, 0, 0, 255),
# 	"1": (111, 78, 55, 255),
# 	"2": (200, 160, 10, 255),
# }
# input_str = """
# 000000000
# 000000000
# 000111000
# 001101100
# 000222000
# 000222000
# 000000000
# 000000000
# 000000000
# """
# N = 3
# ROTATION = 0
# MIRRORING_HORZ = 0
# MIRRORING_VERT = 0


# COLORS = {
# 	"0": (111, 78, 55, 255),
# 	"1": (0, 0, 0, 255),
# 	"2": (255, 255, 255, 255)
# }
# input_str = """
# 000000000
# 000212100
# 000121200
# 000212100
# 000121200
# 000000000
# """
# N = 2
# ROTATION = 0
# MIRRORING_HORZ = 0
# MIRRORING_VERT = 0


COLORS = {
	"1": (255, 255, 255, 255),
	"0": (0, 0, 0, 255),
}
input_str = """
111
101
100
100
"""
N = 2
ROTATION = 1
MIRRORING_HORZ = 1
MIRRORING_VERT = 1



OUTPUT_X = 80
OUTPUT_Y = 80




def run(referenceGlobal, OX:int, OY:int):
	main(referenceGlobal, input_str, N, ROTATION, MIRRORING_HORZ, MIRRORING_VERT, OX, OY, COLORS)

# referenceGlobal, N, ROTATION, MHORZ, MIRRORING_VERT, OUTPUT_X, OUTPUT_Y
def main(referenceGlobal, IS:str, N:int, R:bool, MH:bool, MV:bool, OX:int, OY:int, c:dict[tuple[int,int,int,int]]):
	input_str, ROTATION, MIRRORING_HORZ, MIRRORING_VERT, OUTPUT_X, OUTPUT_Y, COLORS = IS, R, MH, MV, OX, OY, c

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

	tiles = set()

	block_types = set(i for row in input_map for i in row)
	block_count = len(block_types)
	block_type_from_id = {b: a for a, b in zip(list(block_types), range(block_count))}
	block_id_from_type = {block_type_from_id[a]: a for a in block_type_from_id}

	print(block_type_from_id)
	# print(block_id_from_type, block_type_from_id, sep='\n')

	for x in range(input_width - N + 1):
		for y in range(input_height - N + 1):
			t = tuple(tuple(block_id_from_type[k] for k in i[x:x+N]) for i in input_map[y:y+N])

			if not t in tiles:
				tiles.update(createRotations(t))
				
	tile_map = {i: [k for k in tiles if any(i in row for row in k)] for i in range(block_count)}

	print(f"Generated {len(tiles)} Tiles")
	# print(*tiles, sep='\n')

	wave = [[[1 for _ in range(block_count)] for _ in range(OUTPUT_X)] for _ in range(OUTPUT_Y)]

	def lowestEntropy(wave:list[list[list[bool]]]) -> tuple[int,int]:
		
		# finds the least entropy count of the wave by summing true indices that are non-zero
		# use non-zero outputs bc zero == complete
		solved = all(sum(i) == 1 for row in wave for i in row)
		
		if solved: return 0, [], 0, False, True

		lowest_count = min(a for row in wave for i in row if (a:=sum(i)) != 1)
		
		failed = lowest_count == 0

		if failed: return 0, [], 0, True, False

		# Scoring
		score = [[0 for _ in range(OUTPUT_X)] for _ in range(OUTPUT_Y)]

		for x in range(OUTPUT_X):
			for y in range(OUTPUT_Y):
				score[y][x] = sum(wave[y][x])

		# Output
		tileScore = [[[0 for _ in range(block_count)] for _ in range(OUTPUT_X)] for _ in range(OUTPUT_Y)]
		boxCount = [[0 for _ in range(OUTPUT_X)] for _ in range(OUTPUT_Y)]

		for box_x in range(OUTPUT_X - N + 1):
			# box is top left corner of box, rel is relative location of filled point
			for box_y in range(OUTPUT_Y - N  + 1):
				
				for tile in tiles:
					# Check filled point first, then check all points
					if all(all(wave[box_y + oy][box_x + ox][tile[oy][ox]] for ox in range(N)) for oy in range(N)):
						# print(tile)
						for oy in range(N):
							for ox in range(N):
								tileScore[box_y+oy][box_x+ox][tile[oy][ox]] += 1
				for oy in range(N):
					for ox in range(N):
						boxCount[box_x+oy][box_y+ox] += len(tiles)
		# Find minimum location
		leastTiles = float('inf')
		for x in range(OUTPUT_X):
			for y in range(OUTPUT_Y):
				total = round(sum(tileScore[y][x]) / boxCount[y][x], 3)
				if total < leastTiles and score[y][x] != 1: 
					leastTiles = total
					
		pos = [(x,y) for x in range(OUTPUT_X) for y in range(OUTPUT_Y) if round(sum(tileScore[y][x]) / boxCount[y][x], 3) == leastTiles and score[y][x] != 1]
		
		# print(minLoc, leastTiles)

		# escore = [[0 for _ in range(OUTPUT_X)] for _ in range(OUTPUT_Y)]
		
		# for x in range(OUTPUT_X):
		# 	for y in range(OUTPUT_Y):
		# 		total = 0
		# 		count = 0
		# 		for dx in range(- N + 1, N):
		# 			if not (0 <= x + dx < OUTPUT_X): continue
		# 			for dy in range(-N + 1, N):
		# 				if not (0 <= y + dy < OUTPUT_Y): continue
		# 				total += score[y + dy][x + dx]
		# 				count += 1
		# 		escore[y][x] = round(total / count, 2)

		# lowest_escore = min(escore[y][x] for x in range(OUTPUT_X) for y in range(OUTPUT_Y) if (score[y][x]) != 1)

		# for x in range(OUTPUT_X):
		# 	for y in range(OUTPUT_Y):
		# 		if escore[y][x] == lowest_escore and score[y][x] != 1:
		# 			pos.append((x,y))
		
		# print(lowest_escore)

		x, y = choice(pos)
		return x, y, tileScore[y][x], False, False

	def update(wave:list[list[list[bool]]], point:tuple[int, int], filled:int) -> list[tuple[int, int]]:
		x, y = point

		# Output
		update_block = [[[0 for _ in range(block_count)] for _ in range(2*N-1)] for _ in range(2*N - 1)]

		for dx in range(N):
			# box is top left corner of box, rel is relative location of filled point
			box_x = x - N + 1 + dx
			rel_x = N - dx - 1

			if box_x < 0 or box_x + N > OUTPUT_X: continue
			for dy in range(N):
				box_y = y - N + 1 + dy
				rel_y = N - dy - 1
				if box_y < 0 or box_y + N > OUTPUT_Y: continue
				
				for tile in tile_map[filled]:
					# Check filled point first, then check all points
					if tile[rel_y][rel_x] != filled: continue
					if all(all(wave[box_y + oy][box_x + ox][tile[oy][ox]] for ox in range(N)) for oy in range(N)):
						# print(tile)
						for oy in range(N):
							for ox in range(N):
								update_block[dy+oy][dx+ox][tile[oy][ox]] = 1
			
				# print(box_x, box_y, rel_x, rel_y)

		queue = []
		for oy in range(2*N - 1):
			pos_y = y - N + 1 + oy
			if pos_y < 0 or pos_y >= OUTPUT_Y: continue
			for ox in range(2*N - 1):
				pos_x = x - N + 1 + ox
				if pos_x < 0 or pos_x >= OUTPUT_X: continue
				if sum(wave[pos_y][pos_x]) != 1 and sum(update_block[oy][ox]) == 1: queue.append((pos_x, pos_y))
				wave[pos_y][pos_x] = update_block[oy][ox]
		
		
		# print()
		# print(*wave, sep="\n\n")

		return queue

	def solve(inputWave:list[list[list[bool]]]):
		# Choose random place with lowest entry

		x, y, prob, failed, solved = lowestEntropy(inputWave)
		
		if failed: return
		if solved: return inputWave

		printgraph(inputWave)
		# print("-----")

		indicies = [i for i in range(block_count) if inputWave[y][x][i]]
		weights = [prob[i] for i in range(block_count) if inputWave[y][x][i]]
		shuffle(indicies)

		while indicies:
			if sum(weights) <= 0:
				break
			fill = choices(indicies, weights)[0]

			index = indicies.index(fill)
			weights.pop(index)
			indicies.pop(index)

			wave = deepcopy(inputWave)

			wave[y][x] = [0 if i != fill else 1 for i in range(block_count)]

			# print("Filling", x, y, fill)
			filled = -1
			single_queue = [(x, y)]
			while single_queue:
				pos = single_queue.pop()
				
				filled = -1
				for i in range(block_count): 
					if wave[pos[1]][pos[0]][i]:
						filled = i
						break
				if filled == -1: break

				single_queue.extend(update(wave, pos, filled))	
			
			if filled == -1: continue

			ret = solve(wave)
			
			if ret != None: return ret

	def printgraph(wave):
		output_colors = [[[0, 0, 0, 0] for _ in range(OUTPUT_X)] for _ in range(OUTPUT_Y)]
		for y in range(len(wave)):
			for x in range(len(wave[y])):
				current = wave[y][x]
				totals = [0, 0, 0, 0]
				count = 0
				for i in range(len(current)):
					if current[i]:
						for k in range(len(totals)):
							totals[k] += COLORS[block_type_from_id[i]][k]
						count += 1
				if count != 0:
					totals = tuple(map(lambda x: x//count, totals))
				else:
					totals = tuple(totals)
				output_colors[y][x] = totals
		referenceGlobal[:] = output_colors
		# referenceGlobal.extend(deepcopy(wave))
		# for row in wave:
		# 	for i in range(len(row)):
		# 		for k in range(block_count):
		# 			# print(row)
		# 			if sum(row[i]) != 1:
		# 				print("X", end="")
		# 				break
		# 			if row[i][k] == 1:
		# 				print(block_type_from_id[k], end="")
		# 				break
		# 	print()

	wave = solve(wave)
	printgraph(wave)



if __name__ == "__main__":
	run()