from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import numpy as np
from skimage import io, transform




class SuperMarketDataset(Dataset):
    """Supermarket products dataset"""
    TRAIN_RATIO=0.6
    VAL_RATIO=0.2
    #The different types of datasets possible:
    TRAIN='train'
    VALIDATION='validation'
    TEST='test'
    TYPES_DATASETS=[TRAIN,VALIDATION,TEST]

    def __init__(self, root_dir='.', transform=None,type='train'):
        """
        Args:
            root_dir (string): Directory with all the subfolders to the different products.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.type=type
        self.transform=transform

        ENCODING_TO_CLASSES_FILE_NAME='encoding2classes.txt'
        CSV_FILE_NAME='bb.csv'
        


        def add_data_from_class(data,className,oneHotEncoding):
            fname_with_path=os.path.join(className,CSV_FILE_NAME)
            data[className]=pd.read_csv(fname_with_path)
            return

        def transform_data_according_to_type(self):
            for key in self.data.keys():
                current=self.data[key]
                if self.type==self.TRAIN:
                    self.data[key]=current.iloc[0:round(len(current)*self.TRAIN_RATIO)]
                elif self.type==self.VALIDATION:
                    self.data[key]=current.iloc[round(len(current)*self.TRAIN_RATIO):round(len(current)*(self.TRAIN_RATIO+self.VAL_RATIO))]
                elif self.type==self.TEST:
                    self.data[key]=current.iloc[round(len(current)*(self.TRAIN_RATIO+self.VAL_RATIO)):]






        classNames=[]
        for fileName in os.listdir():
            if os.path.isdir(fileName):
                classNames.append(fileName)
        #Now I will output the file with the classes one hot encoding and names
        numberOfClasses=len(classNames)
        self.classNamesToEncodings={} #Here I will save my classes to one hot encodings in both ways
        #I will also have a data datastructure where data[classname]=its dataframe.
        self.data={}

        classIndex=0
        for className in classNames:
            oneHotEncoding=np.zeros(numberOfClasses)
            oneHotEncoding[classIndex]=1
            self.classNamesToEncodings[className]=oneHotEncoding
            self.classNamesToEncodings[str(oneHotEncoding)]=className
            add_data_from_class(self.data,className,oneHotEncoding)
            classIndex+=1
        transform_data_according_to_type(self)

    def __len__(self):
        total_num_samples=0
        for key in self.data.keys():
            total_num_samples+= len(self.data[key])



        return total_num_samples

    def __getitem__(self, idx):
        #iterate over the dataset until we reach the idxth example
        for className in self.data.keys():
            if len(self.data[className])>=idx+1:
                sample=self.data[className].iloc[idx]
                if self.transform:
                    sample = self.transform(sample)
                fullImagePath=os.path.join(className,sample[0]) #sample is a pandas series which has the image name and the bounding box
                img=io.imread(fullImagePath)
                return {'className':className,'img':img,'bb':sample[1:]}
            idx-=len(self.data[className])
        raise IndexError() #if we are here then there's no such sample in our dataset.


'''
Some convenience methods for debugging purposes. To be moved as soon as debugging is over.
'''
import matplotlib.pyplot as plt
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
s=SuperMarketDataset(type='all')
def show_sample(sample):
    img=sample['img']
    bounding_box= tuple(sample['bb'])
    draw_bounding_box(img,bounding_box)
    plt.imshow(img)
    plt.title(sample['className'])
    plt.show()

def show_samples(a,b,dataset=s):
    for sample_index in range(a,b):
        plt.subplot(1,b-a,sample_index-a+1)
        sample=dataset[sample_index]
        img=sample['img']
        bounding_box= tuple(sample['bb'])
        draw_bounding_box(img,bounding_box)
        plt.imshow(img)
        plt.title(sample['className'])
    plt.show()
