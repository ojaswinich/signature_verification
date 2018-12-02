import tensorflow as tf
import keras
from keras.layers import Input, Lambda,subtract,GlobalMaxPooling2D,Dense,GlobalAveragePooling2D, concatenate, Activation
from keras.applications.mobilenet import MobileNet as Net
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.models import Model
from keras.layers import Input, merge
from keras.optimizers import Adam
import cv2
import numpy as np
import os
from PIL import Image
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input,Lambda,subtract,GlobalMaxPooling2D,Dense,GlobalAveragePooling2D,concatenate,Activation
from keras.applications.xception import Xception as Net
from keras.preprocessing import image
from keras.applications.xception import preprocess_input
from keras.models import Model
ALPHA = 0.7 # Triplet Loss Parameter


# Triplet Loss function
def triplet_loss(x):
    anchor, positive, negative = x
    
    #x = tf.constant([[1, 1, 1], [1, 1, 1]]) ------ tf.reduce_sum(x, 1)  # [3, 3]
    #It is trained in batches so, the tensor will be of the above shape for each triplet
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)
    
    #x = tf.constant([[1, 1, 1], [1, 1, 1]])--------->tf.reduce_sum(x, 0)  # [2, 2, 2]

    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), ALPHA)
    loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)

    return loss


#Deep CNN Model with triplet loss
def create_model_new(d1, d2, c, baseline_model):
    
    # The triplet network takes 3 input images: 2 of the same class and 1 out-of-class sample
    #shape: A shape tuple (integers), not including the batch size. For instance, shape=(32,) 
    #indicates that the expected input will be batches of 32-dimensional vectors.
    
    anchor_example = Input(shape=(d1, d2, c), name = 'anchor')
    positive_example = Input(shape=(d1, d2, c), name = 'positive')
    negative_example = Input(shape=(d1, d2, c), name = 'negative')
        
    base_model = baseline_model
    # the weights of this layer will be set to ones and fixed  (since they
    # are shared we could also leave them trainable to get a weighted sum)
    
    # feed all 3 inputs into the pretrained keras model
    x1 = base_model(anchor_example)
    x2 = base_model(positive_example)
    x3 = base_model(negative_example)
    
    # flatten/summarize the models output:
    # (here we could also use GlobalAveragePooling or simply Flatten everything)
    #Takes the max value for each of the filter
    anchor =   GlobalMaxPooling2D()(x1)
    positive = GlobalMaxPooling2D()(x2)
    negative = GlobalMaxPooling2D()(x3)
    
    
    #Loss gives the final loss value between a,p and n.
    loss = merge([anchor, positive, negative], mode=triplet_loss, output_shape=(1,))

    triplet_model = Model(inputs=[anchor_example, positive_example, negative_example],
                  outputs=loss)
    
    triplet_model.compile(loss='mean_absolute_error', optimizer=Adam())
    print(triplet_model.summary())
    
    return triplet_model

########### FRAMEWORK FUNCTIONS
#Step1----Getting test image
def img_to_array(image):
    img = cv2.resize(image, (128, 128)) 
    x_train = np.array([img])
    return x_train

def img_path_to_array(image_path):
    img1 = cv2.imread(image_path, 1)
    return img_to_array(img1)
 
#Step2----Getting the image embeddings 
def get_triple_embedding(image_path, cnn_model):
    input_img= img_path_to_array(image_path)
    img_embed=cnn_model.predict([input_img,input_img,input_img])
    return img_embed
    
#Step3----Check if image ID is present in the database
def check_if_in_db(database,image_id):
    if image_id in database.keys():
        output='present'
    else :
        output='absent'
    return output
    
#Step4----Add the image to the database if the ID does not exist already 
def add_img_to_db(database,image_id,image_embed):
    database[image_id]=image_embed
    return database
    
#Step5----If the ID exists in the database, take the difference vector of embeddings and 
# predict with logistic model
def predict_img_status(database,image_id, image_embed,log_model):
    anchor_embed=database[image_id]
    test=anchor_embed-image_embed
    pred=log_model.predict(test)
    if (pred==0):
        output='genuine'
    else :
        output='forged'
    return output

# This is the final framework wrapper function   
def final_framework(database, image_path, image_id, cnn_model, log_model):
    display(Image.open(image_path))
    # get image embedddings
    image_embed=get_triple_embedding(image_path, cnn_model)
    # check if image id is present in the database
    check=check_if_in_db(database,image_id)
    # if image is not present, then add to database dictionary
    if (check=='absent'):
        database=add_img_to_db(database,image_id,image_embed)
        print('Added the new ID to the database with embedding.')
    else:
        prediction_output=predict_img_status(database,image_id, image_embed, log_model)
        if (prediction_output=='genuine'):
            print('This is a genuine signature.')
        else:
            print('This is a forged signature.')
    return database
    
    

    
