import pandas as pd
import numpy as np
import scipy.misc as sc
import matplotlib.pyplot as plt
import time
def imshow(img):

    plt.imshow(np.uint8(img))
    time.sleep(0.5)  
    plt.show()
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



CSV_FILE_NAME='bb.csv'
df=pd.read_csv(CSV_FILE_NAME)

for sample_index in range(len(df)):
	current_sample=df.loc[sample_index]
	#translate the sample into image and bounding box
	image_file_name=current_sample[0]
	bounding_box=tuple([p for p in current_sample[1:]])
	img=sc.imread(image_file_name)
	draw_bounding_box(img,bounding_box)
	imshow(img)


