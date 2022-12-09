from time import time

OUTPUT_X = 1000
OUTPUT_Y = 1000

tile_count = 4

wave = [[[1 for _ in range(tile_count)] for _ in range(OUTPUT_X)] for _ in range(OUTPUT_Y)]

wave[0][0][0] = 1
wave[0][0][1] = 0
wave[0][0][2] = 0
wave[0][0][3] = 0

N = 2

tile_array = [
	[[1, 1], [0, 0]],
	[[1, 0], [1, 0]],
	[[0, 0], [1, 1]],
	[[0, 1], [0, 1]]
]

colors_array = [
	[255,0,0,255],
	[0,255,0,255]
]

output = [[[0 for _ in range(4)] for _ in range(OUTPUT_X)] for _ in range(OUTPUT_Y)]


st = time()
for x in range(OUTPUT_X):
	for y in range(OUTPUT_Y):
		
		c = 0
		for t in range(tile_count):
			for dx in range(N):
				box_x = x - dx
				if (box_x < 0): continue
				for dy in range(N):
					box_y = y - dy
					if (box_y < 0): continue
					if not wave[box_y][box_x][t]: continue
					
					color_id = tile_array[t][dy][dx]

					output[y][x][0] += colors_array[color_id][0]
					output[y][x][1] += colors_array[color_id][1]
					output[y][x][2] += colors_array[color_id][2]
					output[y][x][3] += colors_array[color_id][3]
				
					c += 1

		output[y][x][0] //= c
		output[y][x][1] //= c
		output[y][x][2] //= c
		output[y][x][3] //= c
print(time() - st)

# colorCount = [[0 for _ in range(OUTPUT_X)] for _ in range(OUTPUT_Y)]

# for x in range(OUTPUT_X):
# 	for y in range(OUTPUT_Y):
# 		for tileId in wave[y][x]:

# 			for dx in range(N):
# 				if x + dx >= OUTPUT_X: continue
# 				for dy in range(N):
# 					if y + dy >= OUTPUT_Y: continue
					
# 					# we are sorry.
# 					itemColor = colors_array[tile_array[tileId][dy][dx]]
# 					output[y+dy][x+dx] = [output[y+dy][x+dx][i] + itemColor[i] for i in range(4)]
					
# 					colorCount[y+dy][x+dx] += 1
# for x in range(OUTPUT_X):
# 	for y in range(OUTPUT_Y):
# 		output[y][x] = [output[y][x][i] // colorCount[y][x] for i in range(4)]


# print(*output, sep="\n")