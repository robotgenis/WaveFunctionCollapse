import random
import pyglet
from copy import deepcopy
def run(r, OX, OY):

	# Define the size and colors of the grid squares
	size = 10
	
	# Set the window size and background color
	width, height = size*OX, size*OY
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
		
		wave = deepcopy(r)
		
		batch = pyglet.graphics.Batch()

		for y in range(len(wave)):
			for x in range(len(wave[y])):
				ox = x * size
				oy = height - y * size - size
				batch.add(4, pyglet.gl.GL_QUADS, None,
					('v2i', (ox, oy, ox + size, oy, ox + size, oy + size, ox, oy + size)),
					('c4B', wave[y][x] * 4)
				)
	
		# window.clear()
		batch.draw()	


	# Run the window
	pyglet.clock.schedule_interval(update, 0.1)
	pyglet.app.run()