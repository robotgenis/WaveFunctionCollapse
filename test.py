import numpy
import pyglet
from pyglet.gl import *

# the size of our texture
dimensions = (4, 4)

# we need RGBA textures
# which has 4 channels
format_size = 4
bytes_per_channel = 1

# populate our array with some random data
data = numpy.random.random_integers(
    low = 0,
    high = 1,
    size = (dimensions[ 0 ] * dimensions[ 1 ], format_size)
    )

# convert any 1's to 255
data *= 255
        
# set the GB channels (from RGBA) to 0
data[ :, 1:-1 ] = 0
        
# ensure alpha is always 255
data[ :, 3 ] = 255

print(data)

# we need to flatten the array
data.shape = -1



tex_data = (GLubyte * data.size)( *data.astype('uint8') )

print(tex_data)

img = pyglet.image.ImageData(
    dimensions[ 0 ],
    dimensions[ 1 ],
    "RGBA",
    tex_data,
    pitch = dimensions[ 1 ] * format_size * bytes_per_channel
    )



window = pyglet.window.Window(width=dimensions[0], height=dimensions[1])

@window.event
def on_draw():
	img.blit(0, 0)


# Run the window
pyglet.app.run()