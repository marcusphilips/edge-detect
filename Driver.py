# coding=utf-8
from PIL import Image, ImageFile
import numpy as np
import cv2
# do not use multiprocessing.dummy; it doesn't use all the threads
from multiprocessing import Pool
import math
import time

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Block:
    """
    A slap-dashed class to represent an object because fiddling with lists is super annoying.
    """
    id_list = 0

    def __init__(self, sub_block: list):
        """
        Intializes a new column with and implements the sub_block list
        :param sub_block: standard sub_block set up
        """
        self.done = False
        self.block = []
        throwaway = [sub_block[0:-1]]
        self.block.append(throwaway)
        self.ranges = []
        self.ranges.append(sub_block[-1])
        self.block_id = id_list
        id_list += 1

    def identify_block_as_done(self):
        """
        There was no more new columns to add, so mark the block completed so that it won't be messed with again
        :return: nothing
        """
        self.done = True

    def join_block(self, b):
        """
        Joins the contents of the block in the parameter to the implicit Block associated
        :param b:
        :return: nothing
        """
        b.trim()
        addend = b.get_block()
        for col in addend:
            # check if x-coordinates match, if so add as a new sub-block
            # key sub-block -> xy values -> an x or y value
            column_num = -1
            for self_col in range(self.block.__len__()):
                if col[0][0][0] == self.block[self_col][0][0][0]:
                    column_num = self_col
                    break
            # found an existing matching column
            if column_num > -1:
                # for most cases this will only probably loop once
                for sb in col:
                    self.block[column_num].append(sb)
            else:
                # I'll just stick the column at the beginning
                # order isn't super important

                self.block.insert(0, col)

    def get_block(self):
        return self.block

    def is_done(self):
        """
        Is the block done?
        :return: a bool, true if the block is done, or false if not completed
        """
        return self.done

    def get_last_column(self):
        # key block -> column -> sub-block -> xy values -> an x or y value
        # val = self.block[-1][0][0][0]
        return self.block[-1][0][0][0]

    def add_to_if_overlap(self, sb: list):
        """
        Checks if a sub-block can be added with overlapping ranges but does not check if x-values match up, and if so,
        does. Also checks if it is not done which if it is done then it will not add
        :param sb: a sub-block with the last index as reserved as the range
        :return: a bool if the sub_block was added
        """
        if not self.done:
            if self.ranges_intercept(sb[-1]):
                self.block[-1].append(sb[0: -1])
                return True
            else:
                return False
        else:
            return False

    def add_force(self, sb: list):
        """
        Adds a list of pixel values regardless if they actually line up or intercept or regardless of done flag
        :param sb: a the standard sub_block setup
        :return: nothing
        """
        self.block[-1].append(sb[0: -1])

    def has_coordinate(self, coord: tuple) -> bool:
        """Sees if a coordinate already exits in the Block object
        :param coord: a xy coordinate tuple NOT list
        :return: true if an xy coordinate already exists in the Block object, false if otherwise
        """
        for col_val in range(self.block.__len__()):
            for pixel_val in range(self.block[col_val].__len__()):
                if coord == pixel_val:
                    return True
        return False

    def expand_block(self):
        """
        Expands the block. Think of like adding an empty column. Will not add column if done flag is set to true.
        :return: new size of the block if you want it
        """
        if not self.done:
            self.block.append([])
        return self.block.__len__()

    def reset_ranges(self, degree_of_separation=1):
        """Resets the Block objects ranges to be the last column's sub-block's ranges if it is not done
        :param degree_of_separation: how wide do you want the ranges to be
        :return: nothing
        """
        if not self.done:
            self.ranges = []
            last_column = self.block[-1]
            # if last_column.__len__() != 1:
            for sb in range(last_column.__len__()):
                begin = last_column[sb][0][1] - degree_of_separation
                last = last_column[sb][-1][1] + degree_of_separation
                self.ranges.append((begin, last))

    def ranges_intercept(self, range2: tuple) -> bool:
        """
        Compares the all ranges in the Block object to another tuple/list range
        :param range2: a tuple or list of size 2
        :return: a Boolean if they overlap
        """
        for r in self.ranges:
            if r[0] <= range2[0] <= r[1] or r[0] <= range2[1] <= r[1]:
                return True
        return False

    def get_coordinate(self, column: int, index: int) -> tuple:
        return self.block[column][index]

    def column_height(self, column: int) -> int:
        return self.block[column].__len__()

    def get_array(self):
        """
        Gives you the raw 1D list (not array) of tuple xy coordinates.
        :return: a type list each with a list with an xy tuple
        """
        acc = []
        for column in self.block:
            for sb in column:
                for pixel_val in sb:
                    acc.append(pixel_val)
        return acc

    def trim(self):
        """
        Removes any empty columns at the end
        :return:
        """
        for b in reversed(self.block):
            if b == []:
                self.block.pop()
            else:
                break

    def __len__(self):
        """How many columns are there per a given block"""
        return self.block.__len__()


def get_sub_blocks(col, degree_of_separation=1, tolerance=240):
    column = []
    tem = []
    for row in range(col.__len__() - 1):
        pix = col[row]
        if pix > tolerance:
            tem.append((col[-1], row))
        if tem.__len__() != 0 and pix <= tolerance:
            # append a range tuple at the end
            # wouldn't a class make more sense?
            # maybe, but that feels like more of a can of worms
            # means add a tuple with a first pixel's y value - 1 and with the last pixel's y value + 1
            tem.append((tem[0][1] - degree_of_separation, tem[-1][1] + degree_of_separation))
            column.append(tem)
            # reset sub-block
            tem = []
        # block is highlighted but reached end of height
    if tem.__len__() != 0:
        # means add a tuple with a first pixel's y value - 1 and with the last pixel's y value + 1
        tem.append((tem[0](1) - degree_of_separation, tem[-1](1) + degree_of_separation))
        column.append(tem)
        # each index of the sub-block represents a column
    if column.__len__() != 0:
        return column


def get_indices(data):
    sub_block = data[0: -1]
    active_blocks = data[-1]
    block_intercepts = []
    for bl in range(active_blocks.__len__()):
        if active_blocks[bl].ranges_intercept(sub_block[-1]):
            block_intercepts.append(bl)
    if block_intercepts.__len__() > 0:
        return block_intercepts


def get_points(pixels, tolerance=200, degree_of_separation=1):
    """Takes in 2D list/array and returns pixels that it thinks are relevant for vectorization.
    ---Details---
    It does this by "splicing" pixels in a column (or row) and then creates a "one dimensional" block from the grouped
    pixels. For the next column it also groups the pixels in a similar manner but then checks the ranges of the previous
    sub-blocks and see if the new sub-blocks concur. If so, they're added into the growing sub-block.
    """
    # pixels doesn't know what index it will be so I will manually loop to add it in
    for col in range(pixels.__len__()):
        pixels[col].append(col)
    sub_blocks = pool.map(get_sub_blocks, pixels)
    # loop through to delete empty columns
    sh = 0
    for col in range(sub_blocks.__len__()):
        if sub_blocks[col + sh] is None:
            sub_blocks.pop(col + sh)
            sh -= 1
    # each index in the sub_blocks represents an "active column"
    # each index of that represents the 1D sub_blocks and
    # the last tuple of that sub_block represents the "y range"
    blocks = []
    continuous = True
    # key: sub_blocks[columns][sub_blocks][xy coordinates with last tuple a range][x or y/min and max]
    for col_val in range(sub_blocks.__len__() - 1):
        # if col_val % 20 == 0:
        #     print("done with column " + str(col_val))
        if col_val == 0:
            continuous = False
        elif sub_blocks[col_val][0][0][0] == sub_blocks[col_val - 1][0][0][0] + 1:
            continuous = True
        else:
            continuous = False
            for b in range(blocks.__len__()):
                blocks[b].identify_block_as_done()
        # for each col_val check if any of the current sub_blocks intercept with the next sub_blocks
        # so two nested for loops
        if continuous:
            blocks_to_send = []
            for bl in blocks:
                if not bl.is_done():
                    blocks_to_send.append(bl)
            column_to_send = sub_blocks[col_val]
            for sb in column_to_send:
                column_to_send.append(blocks_to_send)
            indices = pool.map(get_indices, column_to_send)
            for r in range(indices.__len__()):
                # No intercepts
                if indices[r] is None:
                    blocks.append(Block(column_to_send[0: -1]))
                elif indices[r].__len__() == 1:
                    blocks[]
            # for sb in sub_blocks[col_val]:
            #     b_list = []
            #     lost_block = True
            #     for b in range(blocks.__len__()):
            #         if not blocks[b].is_done():
            #             if blocks[b].ranges_intercept(sb[-1]):
            #                 lost_block = False
            #                 b_list.append(b)
            #     if lost_block:
            #         blocks.append(Block(sb))
            #     else:
            #         if b_list.__len__() == 1:
            #             blocks[b_list[0]].add_to_if_overlap(sb)
            #         else:
            #             blocks[b_list[0]].add_to_if_overlap(sb)
            #             for val in range(1, b_list.__len__()):
            #                 blocks[b_list[0]].join_block(blocks[b_list[val]])
            #             shift = 0
            #             for val in range(1, b_list.__len__()):
            #                 blocks.pop(b_list[val] - shift)
            #                 shift += 1
        else:
            for sb in sub_blocks[col_val]:
                blocks.append(Block(sb))
        # make sure each block has a new column value or else get set to done
        for b in range(blocks.__len__()):
            if not blocks[b].is_done():
                blocks[b].trim()
                if blocks[b].get_last_column() != sub_blocks[col_val][0][0][0]:
                    blocks[b].identify_block_as_done()
                blocks[b].reset_ranges()
                blocks[b].expand_block()
    for block_i in range(blocks.__len__()):
        blocks[block_i].trim()
    return blocks


def is_next_to(o, g):
    close = False
    if abs(o[0] - g[0]) == 1:
        close = True
    elif abs(o[1] - g[1] == 1):
        close = True
    return close


def get_pixel(dp):
    # print("called")
    return i.getpixel((dp[0], dp[1]))


def enhance(data) -> float:
    """Takes in a float and inputs into a logistic function (think of an S-curve).
        This makes it so dim values go to basically zero and brighter values are brought up to near full brightness.
    """
    #     dif_rgb_values[x][y] *= multiplier
    #     dif_rgb_values[x][y] = 255.0 / (1 + math.exp(-0.1 * (dif_rgb_values[x][y] - 63.75)))
    return 255.0 / (1 + math.exp(-0.1 * (data[0] * data[1] - 63.75)))


def get_largest_d(data):
    """
    color data of 4 pixels
    With 3 color channels that would be
    12(!) pieces of data to put there
    :param data:
    :return:
    """
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
    biggest_dev = 0
    for d in deviations:
        biggest_dev += d
    biggest_dev /= deviations.__len__()
    return biggest_dev


#
# cap = cv2.VideoCapture(0)
# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#
#     # Our operations on the frame come here
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     # Display the resulting frame
#     cv2.imshow('frame',gray)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()


start_time = time.time()
filename = "small.jpg"
i = Image.open(filename)
# print(i.format, i.size, i.mode)

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
pool = Pool(2)

# I think is a row major
# Actually column major
# I don't know
for x in range(i.width):
    temp = []
    for y in range(i.height):
        temp.append(get_pixel((x, y)))
    rgb_values.append(temp)

# print("Collected RGB values")

# I believe this is column major
for x in range(1, rgb_values.__len__()):
    temp = []
    ls = []
    for y in range(1, rgb_values[0].__len__()):
        # x, y
        ls.append((rgb_values[x][y], rgb_values[x][y - 1], rgb_values[x - 1][y], rgb_values[x - 1][y - 1]))
    temp = pool.map(get_largest_d, ls)
    # if x % 100 == 0:
    #    print("Completed column " + str(x))
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
# print("Found Largest Value")

# Logistic Curve
for count in range(2):
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
# print("Finished enhancing")

i = Image.open(filename)
# now to build the image and
# SEE THE EDGES
e = Image.new(mode="RGBA", size=[dif_rgb_values.__len__(), dif_rgb_values[0].__len__()])
# og ie original gangsta ie original image
og = Image.new(mode="RGBA", size=[dif_rgb_values.__len__(), dif_rgb_values[0].__len__()])

diff_blocks = get_points(dif_rgb_values)

switch = False
switch2 = False
for block in diff_blocks:
    raw_block = block.get_array()
    v = (0, 0, 0, 0)
    if not switch and not switch2:
        v = (0, 255, 0, 255)
        switch = True
    elif switch and not switch2:
        v = (255, 255, 10, 255)
        switch2 = True
        switch = False
    elif not switch and switch2:
        v = (0, 0, 255, 255)
        switch2 = True
        switch = True
    elif switch and switch2:
        v = (255, 0, 255, 255)
        switch = False
        switch2 = False
    for _pixel in raw_block:
        # RGBA
        e.putpixel(_pixel, v)


for row_value in range(dif_rgb_values.__len__()):
    for y_val in range(dif_rgb_values[0].__len__() - 1):
        k = (rgb_values[row_value][y_val][0], rgb_values[row_value][y_val][1], rgb_values[row_value][y_val][2], 255)
        og.putpixel([row_value, y_val], k)

# print("Created new \"edged\" image")
# does not work since updated to MacOS Catalina
# e.show()

# print("Overlaying edged image (magenta) over original")

# shift = Image.new(mode="RGBA", size=[dif_rgb_values.__len__(), dif_rgb_values[0].__len__()])

# print("Diff: " + str(dif_rgb_values.__len__()) + " by " + str(dif_rgb_values[0].__len__()))

# for x in range(dif_rgb_values.__len__()):
#     for y in range(dif_rgb_values[0].__len__()):
#         v = int(dif_rgb_values[x][y])
#         # makes green now
#         p = (0, v, 0, v)
#         shift.putpixel([x, y], p)

composite = Image.alpha_composite(og, e)
# does not work since updated to MacOS Catalina
# composite.show()

# the filename is kind of stupid but I don't really care
composite.save("_traced" + filename + ".png")

print("--- %s seconds ---" % (time.time() - start_time))
print(str(diff_blocks.__len__()))
