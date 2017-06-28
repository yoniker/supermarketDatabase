import scipy.misc as sc
import os
import numpy as np
from math import sqrt
import cv2
import matplotlib.pyplot as plt
import time
import copy
IMAGE_EXTENSION='.jpg'

COLORFUL_THRESHOLD=55
CSV_FILE_NAME='bb.csv'

def imshow(img):

    plt.imshow(np.uint8(img))
    time.sleep(0.5)  
    plt.show()



class Silver(object):
        def __init__(self,average_brightness=0,std_brightness=0):
                self.avg=average_brightness
                self.std=std_brightness
        @staticmethod
        def brightness(img):
                fimg=img.astype('float')
                if len(fimg.shape)==1: #if it's actually a pixel
                        return 0.2126*fimg[0] + 0.7152*fimg[1] + 0.0722*fimg[2]
                #otherwise i am going to assume it is an image
                return 0.2126*img[:,:,0] + 0.7152*img[:,:,1] + 0.0722*img[:,:,2]

        @staticmethod
        def brighter_than_neighbors(pixel,neighbors):
                brightness_me=Silver.brightness(pixel)
                for neighbor in neighbors:
                        if Silver.brightness(neighbor)<brightness_me*0.8:
                                return True
                return False


        def __call__(self,pixel):
                the_brightness=Silver.brightness(pixel)
                if the_brightness>self.avg+self.std*2:
                        return True
                return False

def how_colorful(pixel):
        return max(pixel)-min(pixel)


# I will define a colorful pixel as the following:
# a pixel which is colorful,and most of its neighbors are colourful as well.
#For simplicity I will assume that the product is not exactly at the edge of the picture- so no edge pixel is colorful
# Input: img- the image (ndarray)
# xcor,ycor- the x and y cor of the pixel of interest.
# Output: True if it is a colorful pixel (see definition above) and false otherwise
def  is_colorful(img,xcor,ycor):
        imgx,imgy,_=img.shape
        if xcor>=imgx-1 or ycor>=imgy-1 or xcor<=0 or ycor<=0:
                return False
        if silver(img[xcor,ycor,:])==True:
                return True
        neighbors=[img[xcor-1,ycor,:],img[xcor,ycor-1,:],img[xcor+1,ycor,:],img[xcor,ycor+1,:],img[xcor-1,ycor-1,:],img[xcor-1,ycor+1,:],img[xcor+1,ycor-1,:],img[xcor+1,ycor+1,:]]
        if Silver.brighter_than_neighbors(img[xcor,ycor,:],neighbors):
                return True
        colorfulness=np.sum([how_colorful(x)>COLORFUL_THRESHOLD for x in neighbors])
        if colorfulness>len(neighbors)/2:
                return True
        return False

def remove_isolated_points(colorful_points):
    
    for (x,y) in colorful_points:
        num_neighbors=0
        for dx in [-1,0,1]:
            for dy in [-1,0,1]:
                if (x+dx,y+dy) in colorful_points:
                    num_neighbors+=1    
        if num_neighbors<=3: #itself and 2 more.
            colorful_points.remove((x,y))


def get_bounding_box(img,file_img_val='img_values.txt',file_colorful='colorful_values.txt'):
    debug=True
    if debug:
            f_colorful=open(file_colorful,'w')
    imgx,imgy,channels=img.shape
    colorful_points=[]
    for xindex in range(imgx):
        for yindex in range(imgy):
                if is_colorful(img,xindex,yindex):
                        colorful_points.append((xindex,yindex))
                        if debug:
                                f_colorful.write('at '+str(xindex)+":"+str(yindex)+"="+str(img[xindex,yindex,:])+'\n')
    if debug:
        f_colorful.close()
    print('len of colorful points is {}\n'.format(len(colorful_points)))
    remove_isolated_points(colorful_points)
    #product_blob=get_blob_product(img,colorful_points)
    rect_xmin=min([x for (y,x) in colorful_points])
    rect_xmax=max([x for (y,x) in colorful_points])
    rect_ymin=min([y for (y,x) in colorful_points])
    rect_ymax=max([y for (y,x) in colorful_points])
    print("{},{} to {},{}\n".format(rect_xmin,rect_ymin,rect_xmax,rect_ymax))
    color = np.array([255, 0, 0], dtype=np.uint8)
    for (x,y) in colorful_points:
        img[x,y,:]=color
    bounding_box=(rect_xmin,rect_ymin,rect_xmax-rect_xmin,rect_ymax-rect_ymin)
    return bounding_box

def draw_bounding_box(img,bounding_box):
    color = np.array([255, 0, 0], dtype=np.uint8)
    xmin=bounding_box[0]
    ymin=bounding_box[1]
    xmax=xmin+bounding_box[2]
    ymax=ymin+bounding_box[3]
    img[ymin,xmin:xmax] = color
    img[ymax,xmin:xmax] = color
    img[ymin:ymax,xmin] = color
    img[ymin:ymax,xmax] = color

KEEP_ORIGINAL_SIZE=True

MODE='write_csv'
if MODE=='write_csv':
    csv_file=open(CSV_FILE_NAME,'w')
    csv_file.write('img_name,xmin,ymin,width,height\n')
current_file_number=1
files=os.listdir('.')
silver=Silver()
for file in files:
    if len(file)>len(IMAGE_EXTENSION) and file[-len(IMAGE_EXTENSION):]==IMAGE_EXTENSION:
        img=sc.imread(file)
        original_img=copy.deepcopy(img) #save it for now because it will be modified soon
        img=sc.imresize(img, 0.25/2)
        if KEEP_ORIGINAL_SIZE==False:
            original_img=copy.deepcopy(img) #save it for now because it will be modified soon
        brightness=Silver.brightness(img)
        average_brightness=np.average(brightness)
        std_brightness=np.std(brightness)
        silver.avg=average_brightness
        silver.std=std_brightness
        print('img total size is:{}'.format(img.size))
        bounding_box=get_bounding_box(img)
        draw_bounding_box(img,bounding_box)
        if MODE=='show':
            imshow(img)
        print('that was the picture:{}\n'.format(file))
        if MODE=='write_csv':
            sc.toimage(original_img, cmin=0.0, cmax=255.0).save(str(current_file_number)+'.jpg')
            if KEEP_ORIGINAL_SIZE==True:
                bounding_box=tuple(8*x for x in bounding_box) #Rescale back the bounding box.
            csv_file.write(str(current_file_number)+'.jpg,{},{},{},{}\n'.format(*bounding_box))
            current_file_number+=1
if MODE=='write_csv':
    csv_file.close()