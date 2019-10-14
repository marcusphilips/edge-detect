# coding=utf-8
from PIL import Image, ImageFile
# do not use multiprocessing.dummy; it doesn't use all the threads
from multiprocessing import Pool
import numpy as np
import scipy as sp
import math
import time

ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_pixel(dp):
    # print("called")
    return i.getpixel((dp[0], dp[1]))


# ENHANCE with logistic curve so weak spots go gone and medium spots lighten up
# (float, multiplier)
def enhance(data):
    #     dif_rgb_values[x][y] *= multiplier
    #     dif_rgb_values[x][y] = 255.0 / (1 + math.exp(-0.1 * (dif_rgb_values[x][y] - 63.75)))
    return 255.0 / (1 + math.exp(-0.1 * (data[0] * data[1] - 63.75)))


# color data of 4 pixels
# With 3 color channels that would be
# 12(!) pieces of data to put there
def get_largest_d(data):
    # (rgb_values[x][y], rgb_values[x][y-1], rgb_values[x-1][y], rgb_values[x-1][y-1])
    means = []
    # find the sums for each color of the four neighboring
    for color in range(3):
        val = 0
        for pixel in range(4):
            val += data[pixel][color]
        means.append(val)
    for index in range(means.__len__()):
        means[index] = means[index] / 4
    deviations = []
    for index in range(means.__len__()):
        val = 0
        for pixel in range(4):
            val += abs(data[pixel][index] - means[index])
        deviations.append(val / 4)
    biggest_dev = deviations[0]
    # pick largest deviation value to append
    # also I like to use d
    # the x, y, and i 's are getting annoying
    for d in deviations:
        if d > biggest_dev:
            biggest_dev = d
    return biggest_dev


start_time = time.time()
filename = "chair.jpg"
i = Image.open(filename)
print(i.format, i.size, i.mode)

# a 2D grid of 3 RGB values
rgb_values = []

# Between 4 pixels how different are they
# Take the mean of the 4 neighboring rgb values and
# Find the average deviation from the mean
dif_rgb_values = []

# The for loops are hell for my computer
# I thinking about dipping my toes into multi threading
# I finally got multi-threading to work, however,
# It's still balls slow
pool = Pool(4)

# I think is a row major
# Actually column major
# I don't know
for x in range(i.width):
    temp = []
    for y in range(i.height):
        temp.append(get_pixel((x, y)))
    rgb_values.append(temp)

print("Collected RGB values")

# I believe this is column major
for x in range(1, rgb_values.__len__()):
    temp = []
    ls = []
    for y in range(1, rgb_values[0].__len__()):
        # x, y
        ls.append((rgb_values[x][y], rgb_values[x][y-1], rgb_values[x-1][y], rgb_values[x-1][y-1]))
    temp = pool.map(get_largest_d, ls)
    if x % 100 == 0:
        print("Completed column " + str(x))
    temp.append(0)
    dif_rgb_values.append(temp)

temp = []
for k in range(rgb_values.__len__()):
    temp.append(0)
dif_rgb_values.append(temp)

# print("Diff: " + str(dif_rgb_values.__len__()) + " by " + str(dif_rgb_values[0].__len__()))

largest_d = 0
# curious to see what is the largest value
for x in range(dif_rgb_values.__len__()):
    for y in range(dif_rgb_values[0].__len__()):
        try:
            if dif_rgb_values[x][y] > largest_d:
                largest_d = dif_rgb_values[x][y]
        except IndexError:
            print("Index out of range with " + str(x) + " and/or " + str(y))
print("Found Largest Value")

# Logistic Curve
multiplier = 255 / int(largest_d)
big_temp = []
for x in range(dif_rgb_values.__len__()):
    # to add back into dif_rgb_values
    temp = []
    # to send to map
    ls = []
    for y in range(dif_rgb_values[0].__len__()):
        # send in float , multiplier
        ls.append((dif_rgb_values[x][y], multiplier))
    temp = pool.map(enhance, ls)
    big_temp.append(temp)
dif_rgb_values = big_temp
print("Finished enhancing")

i = Image.open(filename)
# now to build the image and
# SEE THE EDGES
e = Image.new(mode="RGBA", size=[dif_rgb_values.__len__(), dif_rgb_values[0].__len__()])
og = Image.new(mode="RGBA", size=[dif_rgb_values.__len__(), dif_rgb_values[0].__len__()])

for x in range(dif_rgb_values.__len__()):
    for y in range(dif_rgb_values[0].__len__()):
        # just gonna show white
        v = int(dif_rgb_values[x][y])
        a = (v, v, v, 255)
        k = (rgb_values[x][y][0], rgb_values[x][y][1], rgb_values[x][y][2], 255)
        # there's probably a faster way to do it gray scale but I don't care
        e.putpixel([x, y], a)
        og.putpixel([x, y], k)
print("Created new \"edged\" image")
e.show()

print("Overlaying edged image (magenta) over original")

shift = Image.new(mode="RGBA", size=[dif_rgb_values.__len__(), dif_rgb_values[0].__len__()])

print("Diff: " + str(dif_rgb_values.__len__()) + " by " + str(dif_rgb_values[0].__len__()))

for x in range(dif_rgb_values.__len__()):
    for y in range(dif_rgb_values[0].__len__()):
        v = int(dif_rgb_values[x][y])
        # makes magenta I believe
        p = (v, 0, v, v)
        shift.putpixel([x, y], p)

composite = Image.alpha_composite(og, shift)
# composite = og
composite.show()

# baseline is 40.13 sec with just the deviation finder being multi-threaded
print("--- %s seconds ---" % (time.time() - start_time))
