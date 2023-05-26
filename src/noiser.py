'''
this adds noisy images to the dataset, though it's not very good at it
i made it in a hurry
'''
import numpy as np
import os
from noise import snoise2
from PIL import Image, ImageDraw
from scipy.spatial import Voronoi

# Configurations
image_num = 100  # The number of images to generate for each type of noise
min_size = 200  # Minimum size of image
max_size = 700  # Maximum size of image

# Directories
output_dir = "dset/train/not_cat"
os.makedirs(output_dir, exist_ok=True)

# Generate images
for i in range(image_num):

    # Random size
    img_sizex = np.random.randint(min_size, max_size)
    img_sizey = np.random.randint(min_size, max_size)

    # Random Noise
    img_random = Image.fromarray(np.uint8(np.random.rand(img_sizex, img_sizey)*255))
    img_random.save(os.path.join(output_dir, f'random_noise_{i}.jpg'))

    # Perlin Noise
    perlin_noise = np.zeros((img_sizex, img_sizey))
    x_idx = np.linspace(0, 1, img_sizey) + np.random.random()*20  # adding a random offset
    y_idx = np.linspace(0, 1, img_sizex) + np.random.random()*20  # adding a random offset
    for j, x in enumerate(x_idx):
        for k, y in enumerate(y_idx):
            perlin_noise[k, j] = snoise2(x, y, octaves=int(6+(np.random.random()*10)), persistence=0.5)  # more octaves and persistence for complex noise
    img_perlin = Image.fromarray(np.uint8((perlin_noise+0.5)*255))  # +0.5 to make noise range from 0 to 1
    img_perlin.save(os.path.join(output_dir, f'perlin_noise_{i}.jpg'))

    # Voronoi Noise
    points = np.random.rand(100, 2) * np.array([img_sizex, img_sizey])  # 100 random two-dimensional points scaled to image size
    vor = Voronoi(points)
    img_voronoi = Image.new('L', (img_sizex, img_sizey))  # 'L' mode for 8-bit pixels, black and white
    
    # Draw the Voronoi diagram on the image
    draw = ImageDraw.Draw(img_voronoi)
    for region in vor.regions:
        if not -1 in region and len(region) > 0:
            polygon = [(vor.vertices[i][0], vor.vertices[i][1]) for i in region]
            # we scale the random color by img_sizex so that larger images have a higher color range
            draw.polygon(polygon, fill=int(np.random.rand()*img_sizex))  
    
    # Save the image
    img_voronoi.save(os.path.join(output_dir, f'voronoi_noise_{i}.jpg'))
