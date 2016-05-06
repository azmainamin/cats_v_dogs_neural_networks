import cPickle as pickle
import numpy as np
import timeit

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv2d

import matplotlib.pyplot as plt
from matplotlib import cm
from sys import version_info
import os
from PIL import Image

import gc

CAT = float(0)
DOG = float(1)
"""
Created on Wed May 04 19:21:48 2016

Preparing the train,  test and validation datasets.

@author: aminm
"""
def pickleToGrayscale(pathname):
    """
        Input: Pickled images.
        Output: Pickled grayscale images separated into 3 batches: train, test, validation.
    """

    batch = pickle.load(open(pathname,'rb'))
    grayscale = []

    #code below rearranges the pixel RGB values. Originally, array is in pattern
    #RGBRGBRGB. code rearranges this so it's all R's first, then G's, then B's.
    #ratchet af
    print "Reading the images..."
    for image in batch:
        stepper = np.arange(0, 3072, 3)
        stepperList = []
        for x in stepper:
            stepperList.append(image[x])
        stepper += 1
        for x in stepper:
            stepperList.append(image[x])
        stepper += 1
        for x in stepper:
            stepperList.append(image[x])
        stepperList = np.array(stepperList)
        
        gray = 0.21*stepperList[0:1024] + 0.72*stepperList[1024:2048] + 0.07*stepperList[2048:3072]
        grayscale.append(gray)

    grayscale = np.asarray(grayscale)
    # Train: 6000 cats + 6000 dogs
    # Test: 4000 cats + 4000 dogs
    # Validation: 1250 cats + 1250 dogs
    train = np.concatenate((grayscale[:6000], grayscale[12500:18500]), axis = 0) 
    test = np.concatenate((grayscale[6000:10000] , grayscale[18500:22500]), axis = 0)
    valid = np.concatenate((grayscale[10000:12500],grayscale[22500:25000]), axis = 0)
    
    with open("32_grayscale_train_img.pkl","wb") as f:
        print "Pickling the training images ..."
        pickle.dump(train, f)

    with open("32_grayscale_test_img.pkl","wb") as f:
        print "Pickling the testing images ..."
        pickle.dump(test, f)

    with open("32_grayscale_valid_img.pkl","wb") as f:
        print "Pickling the validation images ..."
        pickle.dump(valid, f)

    f.close()
def displayImageFromArray(pathname, index):
    """
        Input: A pickled image file.
        Output: Displays the image and label at index
    """
    print "Getting image..."
    labels = pickle.load(open("train_labels.pkl","rb"))
    label = labels[index]
  #  print label
    images = pickle.load(open(pathname,'rb'))
    image = images[index]
    image = Image.fromarray(image.reshape((32,32))) 
    plt.imshow(image,cmap = cm.gray)
    plt.show()

def get_imlist(path):
    """returns a list of filenames for all jpg images in a directory"""
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]

def create_db(img_pkl, labels_pkl, size):
    """
    Input: JPGs from a directory.
    Output: Pickles the resized images and label as np.ndarray in two separate files.
    """
    batch = []
    labels = []

    print "Reading all the images..."
    for file_name in get_imlist(os.getcwd() + "/train"):
       
       if "dog" in file_name:
           label = DOG
       else:
           label = CAT
       imgx = Image.open(file_name)   
       labels.append(label)

       imgx = imgx.resize((size,size))
       imag = np.asarray(imgx)
       imag = np.ravel(imag)
       batch.append(imag)


    #batch = np.asarray(batch)
    labels = np.asarray(labels)
   # print label
    print "Reading images done..."

    #Pickle all 25K images in one pickle file
    with open(img_pkl,"wb") as f:
        print "Pickling the images ..."
        pickle.dump(batch,f,-1)
        gc.collect()
        f.close()
    #Pickle all labels in one file.

    with open("labels.pkl", "wb") as f:
        print "Pickling labels ..."
        pickle.dump(labels,f,-1)

    f.close()
def separate_labels(labels_path):
    """
        Input: a pickled label file
        Output: 3 separate labels for train, test and validation
    """
    
    labels = pickle.load(open(labels_path,'rb'))

    train_label = np.concatenate((labels[:6000],labels[12500:18500]),axis =0)
    test_label = np.concatenate((labels[6000:10000],labels[18500:22500]),axis = 0)
    valid_label = np.concatenate((labels[10000:12500],labels[22500:25000]), axis=0)
    
    with open("train_labels.pkl", "wb") as f:
        print "Pickling train labels ..."
        pickle.dump(train_label,f,-1)
    with open("test_labels.pkl", "wb") as f:
        print "Pickling test labels ..."
        pickle.dump(test_label,f,-1)
    with open("valid_labels.pkl", "wb") as f:
        print "Pickling test labels ..."
        pickle.dump(valid_label,f,-1)

    f.close()
def load_data():
    """
        Loads pickled images and their corresponding labels and returns a list of tuples of Theano
        shared variables.
    """

    train_images = pickle.load(open("32_grayscale_train_img.pkl", "rb"))
   # print len(train_images)
    train_labels = pickle.load(open("train_labels.pkl", "rb"))
 #   print len(train_labels)
    
    valid_images = pickle.load(open("32_grayscale_valid_img.pkl", "rb"))
    valid_labels = pickle.load(open("valid_labels.pkl", "rb"))
    
    
    test_images = pickle.load(open("32_grayscale_test_img.pkl", "rb"))
    test_labels = pickle.load(open("test_labels.pkl", "rb"))
    
    train_set = (train_images, train_labels)
    valid_set = (valid_images, valid_labels)
    test_set = (test_images, test_labels)

    #return train_set, valid_set, test_set


    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')
    
    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

    
def main():
    '''
    This little test script will display the first image, its name, and its target value
    '''
    
    create_db("images_32.pkl", "labels.pkl", 32)
    separate_labels("labels.pkl")
    DATADIR = os.getcwd()
    pickleToGrayscale("images_32.pkl")
    
    displayImageFromArray("32_grayscale_test_img.pkl", 7999)

if __name__ == '__main__':
    main()
