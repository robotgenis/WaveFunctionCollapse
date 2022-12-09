import random
import pyglet
from copy import deepcopy
def run(r, OUTPUT_X, OUTPUT_Y):

	# Define the size and colors of the grid squares in pixels
	size = 5
	
	# Set the window size and background color
	width, height = size*OUTPUT_X, size*OUTPUT_Y

	bg_color = (0, 0, 0, 255)

	# Create a window and set its background color
	window = pyglet.window.Window(width=width, height=height)
	pyglet.gl.glClearColor(*bg_color)


	# colors = [ # toothpaste pallete
	# 	( 50,  41,  47, 255),
	# 	(224, 101, 101, 255),
	# 	(240, 247, 244, 255),
	# 	(153, 225, 217, 255),
	# 	(112, 171, 175, 255),
	# 	(112,  93,  86, 255)
	# ]

	# colors = [ # plague pallete
	# 	( 16,  11,   0, 255),
	# 	(133, 203,  51, 255),
	# 	(239, 255, 200, 255),
	# 	(165, 203, 195, 255),
	# 	( 59,  52,  31, 255),
	# 	( 75, 107,  26, 255)
	# ]	

	colors = [ # pink pallete
		(255, 185, 151, 255),
		(246, 126, 125, 255),
		(132,  59,  98, 255),
		( 11,   3,  45, 255),
		( 64,  44,  76, 255),
		(116,  84, 106, 255)
	]

	# colors = [ # r.g.b.c.y.m.w
	# 	(255,   0,   0, 255),
	# 	(  0,   0, 255, 255),
	# 	(  0, 255,   0, 255),
	# 	(  0, 179, 179, 255),
	# 	(255,   0, 255, 255),
	# 	(255, 255,   0, 255),
	# 	(255, 255, 255, 255),
	# ]

	random.shuffle(colors)

	def update(dt):
		if r == None: return
		if len(r) == 0: return
		
		wave, N, COLORS, tile_type_from_id, block_type_from_id = deepcopy(r)

		colorTotals = [[[0, 0, 0, 0] for _ in range(OUTPUT_X)] for _ in range(OUTPUT_Y)]
		colorCount = [[0 for _ in range(OUTPUT_X)] for _ in range(OUTPUT_Y)]
		
		for x in range(OUTPUT_X):
			for y in range(OUTPUT_Y):
				for tileId in wave[y][x]:
					contents = tile_type_from_id[tileId]

					for dx in range(N):
						if x + dx >= OUTPUT_X: continue
						for dy in range(N):
							if y + dy >= OUTPUT_Y: continue
							
							# we are sorry.
							itemColor = COLORS[block_type_from_id[contents[dy][dx]]]
							colorTotals[y+dy][x+dx] = [colorTotals[y+dy][x+dx][i] + itemColor[i] for i in range(4)]
							
							colorCount[y+dy][x+dx] += 1

		for x in range(OUTPUT_X):
			for y in range(OUTPUT_Y):
				if colorCount[y][x]:
					colorTotals[y][x] = tuple(int(i / colorCount[y][x]) for i in colorTotals[y][x])
				else:
					colorTotals[y][x] = (0,0,0,0)
		
		batch = pyglet.graphics.Batch()

		for y in range(len(colorTotals)):
			for x in range(len(colorTotals[y])):
				ox = x * size
				oy = height - y * size - size
				batch.add(4, pyglet.gl.GL_QUADS, None,
					('v2i', (ox, oy, ox + size, oy, ox + size, oy + size, ox, oy + size)),
					('c4B', colorTotals[y][x] * 4)
				)
	
		# window.clear()
		batch.draw()	


	# Run the window
	pyglet.clock.schedule_interval(update, 0.1)
	pyglet.app.run()