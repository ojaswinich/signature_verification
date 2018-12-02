# Signature Verification
## Objective 
Automated detection of new signatures into genuine or fraud based on one base signature against a unique id (one shot learning).

## Methodology 
We have used a deep CNN architecture coupled with triplet loss to train the signature images. Once the CNN encodings are obtained, we arrange the data in pairs of genuine-genuine or genuine-fraud along with their corresponding labels and fit a logistic model. 

These two final models are used to predict any new image signature into genuine or fraud by comparing it to its base signature.

## Documentation
All the detailed documentation of approach and the accuracy obtained is present in the documents folder. proposal_stage1 contains the approach proposed in stage 1 of the hackathon. proposal_stage2 contains the detailed approach applied here.

## Data Sources
The data used for training the CNN model and the logistic model is from SigComp2009 (train and test) and SigComp2011 (Dutch) data set. A subset of the data was used for training and validation was done on the remaining unseen part of the data. 

## Relevant Code 
All the code for training the CNN model as well as the subsequent logistic model is present in train.ipynb
fn_utils.py contains all the functions that are called for the final framework of one shot learning.
final_framework.ipynb contains the final framework that is built to test any signature image. It loads the CNN and logistic model weights and creates an empty database. 

Whenever a new signature path and ID is given as input to the final function (final_framework), it saves the embedding of the image against the ID in the database and displays the message 'Added the new ID to the database with embedding.'

Now whenever another signature image is passed with the same ID (any ID already present in the databse), it compiles the embedding of this image using the CNN model weights and takes the difference vector of this embedding and the corresponding embedding of the image with this ID present in the database. 

This difference vector is finally passed through the pre-trained logistic model which determines whether the new image is genuine or fraud.

If the prediction output is genuine class, then the framework will display 'This is a genuine signature.'
If the prediction output is forged class, then the framework will display 'This is a forged signature.'

## API
A spyre API can be built on top of this framework which will act as the UI for the signature verification process. 

## Authors
All methodology and code is ideated and authored by:
1. Ojaswini Chhabra
2. Souradip Chakraborty
