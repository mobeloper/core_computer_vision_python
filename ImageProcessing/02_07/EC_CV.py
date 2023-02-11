import numpy as np
import matplotlib.pyplot as plt

# To convert an RGBA image array that represents data with floating 
# point numbers from 0 to 1 into the RGB integer format from 0 to 255, 
# we need to make 4 changes:

# 1) Get rid of the A channel
# 2) Multiply by 255
# 3) Round the resulting values
# 4) Ensure values are between 0 and 255
# 5) Convert data to 8-bit integers

# Let's define a function for this:

def adapt_PNG(the_PNG):
    the_PNG = the_PNG[:,:,:3]      # Step 1
    the_PNG = the_PNG * 255        # Step 2
    the_PNG = adapt_image(the_PNG) # Steps 3, 4, 5
    return the_PNG


# This function will be useful to perform steps 3, 4, and 5 for images
# undergoing operations that may result in floating point numbers. 
def adapt_image(the_img):
    return np.uint8(np.clip(the_img.round(),0,255)) # Steps 3, 4, 5

# This function converts a grayscale image to black and white 
# using a global threshold.
def grayscale_to_BW(grayscale_pic, threshold):
    rows, cols = np.shape(grayscale_pic)
    BW_pic = np.zeros((rows,cols))
    for i in range(rows):
        for j in range(cols):
            BW_pic[i,j] = 0 if grayscale_pic[i,j] <= threshold else 255
    return BW_pic
    
# This function downscales an incoming picture by a linear factor of n.
# This yields a picture that's n^2 smaller (in area) than the original.
def downscale(pic,n):
    rows, cols, temp = np.shape(pic)
    rows = int(n * int(rows/n)) # make sure rows are divisible by n
    cols = int(n * int(cols/n)) # make sure cols are divisible by n
    pic = pic[:rows,:cols]
    rows = int(rows/n)
    cols = int(cols/n)
    smaller = np.zeros((rows,cols,3),np.float64)
    for i in range(rows):
        for j in range(cols):
            smaller[i,j,0] = np.average(pic[i*n:i*n+n,j*n:j*n+n,0])  #    Red Channel
            smaller[i,j,1] = np.average(pic[i*n:i*n+n,j*n:j*n+n,1])  # Green Channel
            smaller[i,j,2] = np.average(pic[i*n:i*n+n,j*n:j*n+n,2])  # Blue Channel
    return smaller


# This function upscales an incoming picture by a linear factor of 2.
# This yields a picture that's 4 times larger (in area) than the original.
def upscale_by_2(pic):
    rows, cols, temp = np.shape(pic)
    last_row = pic[rows-1]
    pic = np.insert(pic, -1, last_row, axis=0)
    last_col = pic[:,cols-1]
    pic = np.insert(pic, -1, last_col, axis=1)+0.0
    larger = np.zeros((rows*2,cols*2,3),np.float64)
    for i in range(rows):
        il = i * 2
        for j in range(cols):
            jl = j * 2
            larger[il,jl] = pic[i,j]  # top-left pixel
            
            # top-right pixel:
            larger[il,jl+1,0] = (pic[i,j,0]+pic[i,j+1,0])/2  #    Red Channel
            larger[il,jl+1,1] = (pic[i,j,1]+pic[i,j+1,1])/2  # Green Channel
            larger[il,jl+1,2] = (pic[i,j,2]+pic[i,j+1,2])/2  # Blue Channel
            
            # bottom-left pixel:
            larger[il+1,jl,0] = (pic[i,j,0]+pic[i+1,j,0])/2  #    Red Channel
            larger[il+1,jl,1] = (pic[i,j,1]+pic[i+1,j,1])/2  # Green Channel
            larger[il+1,jl,2] = (pic[i,j,2]+pic[i+1,j,2])/2  # Blue Channel
            
            # bottom-right pixel:
            larger[il+1,jl+1,0] = (pic[i,j,0]+pic[i+1,j+1,0])/2  #    Red Channel
            larger[il+1,jl+1,1] = (pic[i,j,1]+pic[i+1,j+1,1])/2  # Green Channel
            larger[il+1,jl+1,2] = (pic[i,j,2]+pic[i+1,j+1,2])/2  # Blue Channel

    return larger


# This function calculates a seam from a squared difference matrix.
# The algorithm is similar to Dijkstra's algorithm, with the difference 
# that it expands all nodes at each vertical level.

def get_seam(dif):
    height, width = dif.shape
    cost = dif.copy()
    seam = np.zeros((height))
    
    # Calculate the path costs by traversing the difference top down.
    # For every cost at level j, add the pixel value to the minimum cost from its 
    # 2 or 3 neighbors at lejel j-1 (above).
    for j in range(height):
        if(j == 0):
            continue
        for i in range(width):
            if(i == 0):
                cost[j,i] += np.amin((cost[j-1,i],cost[j-1,i+1]))
                continue
            if(i == width-1):
                cost[j,i] += np.amin((cost[j-1,i-1],cost[j-1,i]))
                continue
            cost[j,i] += np.amin((cost[j-1,i-1],cost[j-1,i],cost[j-1,i+1]))
    
    # Produce the seam by traversing the cost array, picking the lowest-cost
    # elements, following a continuous path.
    for j in reversed(range(height)):
        if(j == height-1):
            seam[j] = np.argmin((cost[j]))
            continue
        down = int(seam[j+1])
        if(down == 0):
            seam[j] = (down) + np.argmin((cost[j,down],cost[j,down+1]))
            continue
        if(down == width-1):
            seam[j] = (down-1) + np.argmin((cost[j,down-1],cost[j,down]))
            continue
        seam[j] = (down-1) + np.argmin((cost[j,down-1],cost[j,down],cost[j,down+1]))
    return seam    
