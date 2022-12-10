import pyglet
from copy import deepcopy
from time import sleep, time

import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
from gpu_helper import createsBlockGridSizes

# Create a CUDA kernel
compute_colors_module = SourceModule("""
__global__ void compute_colors(bool *wave, unsigned int *OUTPUT_X, unsigned int *OUTPUT_Y, unsigned char *N, unsigned int *tile_count, unsigned char *tile_array, unsigned char *colors_array, unsigned int *output) {
	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;
	const int chunk = x + y * *OUTPUT_X;

	if(x >= *OUTPUT_X || y >= *OUTPUT_Y) return;

	output[4 * chunk + 0] = 0;
	output[4 * chunk + 1] = 0;
	output[4 * chunk + 2] = 0;
	output[4 * chunk + 3] = 0;
	
	int counter = 0, t, dx, dy, box_x, box_y;
	for(t = 0; t < *tile_count; ++t){
		for(dx = 0; dx < *N; ++dx) {
			
			box_x = x - dx;
			
			// if box_x is out of bounds, skip
			if(box_x < 0) continue;
			
			for(dy = 0; dy < *N; ++dy){
			
				box_y = y - dy;

				// if box_y is out of bounds, skip
				if (box_y < 0) continue;

				int wave_pos = t + (box_x + box_y * *OUTPUT_X) * *tile_count;
				
				// check if tile exists
				if(!wave[wave_pos]) continue;

				int tile_pos = dx + dy * (*N) + t * (*N) * (*N);	

				int color_id = tile_array[tile_pos];

				output[4 * chunk + 0] += colors_array[4 * color_id + 0];
				output[4 * chunk + 1] += colors_array[4 * color_id + 1];
				output[4 * chunk + 2] += colors_array[4 * color_id + 2];
				output[4 * chunk + 3] += colors_array[4 * color_id + 3];
				
				++counter;
			}
		}
	}
	
	if (counter == 0) counter = 1;
	
	output[4 * chunk + 0] /= counter;
	output[4 * chunk + 1] /= counter;
	output[4 * chunk + 2] /= counter;
	output[4 * chunk + 3] /= counter;
}
""")
compute_colors = compute_colors_module.get_function("compute_colors")

def run(r, OUTPUT_X, OUTPUT_Y):

	# Define the size and colors of the grid squares in pixels
	size = 5
	
	# Set the window size and background color
	width, height = size*OUTPUT_X, size*OUTPUT_Y

	bg_color = (0, 0, 0, 255)

	# Create a window and set its background color
	window = pyglet.window.Window(width=width, height=height)
	pyglet.gl.glClearColor(*bg_color)


	# Transfer the data to the GPU
	while len(r) < 5:
		sleep(0.05)

	wave, N, tile_count, tile_array, colors_array = deepcopy(r)

	output = np.zeros((OUTPUT_Y, OUTPUT_X, 4), dtype=np.uint32)

	out_x = np.uint32(OUTPUT_X)
	out_y = np.uint32(OUTPUT_Y)

	out_x_gpu = drv.mem_alloc(out_x.nbytes)
	out_y_gpu = drv.mem_alloc(out_y.nbytes)
	wave_gpu = drv.mem_alloc(wave.nbytes)
	N_gpu = drv.mem_alloc(N.nbytes)
	tile_count_gpu = drv.mem_alloc(tile_count.nbytes)
	tile_array_gpu = drv.mem_alloc(tile_array.nbytes)
	colors_array_gpu = drv.mem_alloc(colors_array.nbytes)
	output_gpu = drv.mem_alloc(output.nbytes)

	drv.memcpy_htod(out_x_gpu, out_x)
	drv.memcpy_htod(out_y_gpu, out_y)
	drv.memcpy_htod(wave_gpu, wave)
	drv.memcpy_htod(N_gpu, N)
	drv.memcpy_htod(tile_count_gpu, tile_count)
	drv.memcpy_htod(tile_array_gpu, tile_array)
	drv.memcpy_htod(colors_array_gpu, colors_array)

	# data = output.flatten()

	data = []

	for row in output[::-1]:
		temp = []
		for i in row:
			temp.extend(tuple(i)*size)

		data.extend(tuple(temp)*size)

	data = np.array(data).flatten()

	tex_data = (pyglet.gl.GLubyte * data.size)( *data.astype('uint8') )

	format_size = 4
	bytes_per_channel = 1
	img_pitch = width * format_size * bytes_per_channel


	img = pyglet.image.ImageData(
		width,
		height,
		"RGBA",
		tex_data,
		pitch = img_pitch
    )

	@window.event
	def on_draw():
		img.blit(0, 0)

	def update(dt):
		if r == None: return
		if len(r) == 0: return
				
		drv.memcpy_htod(wave_gpu, r[0])

		block, grid = createsBlockGridSizes(OUTPUT_X, OUTPUT_Y, 1)
		
		compute_colors(wave_gpu, out_x_gpu, out_y_gpu, N_gpu, tile_count_gpu, tile_array_gpu, colors_array_gpu, output_gpu, block=block, grid=grid)
		
		# Transfer the result back to the host
		drv.memcpy_dtoh(output, output_gpu)

		# data = output.flatten()

		data = []

		for row in output[::-1]:
			temp = []
			for i in row:
				temp.extend(tuple(i)*size)

			data.extend(tuple(temp)*size)

		data = np.array(data).flatten()

		tex_data = (pyglet.gl.GLubyte * data.size)( *data.astype('uint8') )
		
		img.set_data("RGBA", img_pitch, tex_data)

	# Run the window
	pyglet.clock.schedule_interval(update, 0.5)
	pyglet.app.run()