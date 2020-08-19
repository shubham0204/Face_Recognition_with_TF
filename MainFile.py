
from tensorflow.keras.callbacks import TensorBoard
from SiameseModel import Recognizer
from PIL import Image
import numpy as np
import time

data_dimension = 128

X1 = np.load( 'processed_data/x1.npy')
X2 = np.load( 'processed_data/x2.npy')
Y = np.load( 'processed_data/y.npy')

X1 = X1.reshape( ( X1.shape[0]  , data_dimension**2 * 3  ) ).astype( np.float32 )
X2 = X2.reshape( ( X2.shape[0]  , data_dimension**2 * 3  ) ).astype( np.float32 )

print( X1.shape )
print( X2.shape )
print( Y.shape )

recognizer = Recognizer()
#recognizer.load_model('models/model.h5')

parameters = {
    'batch_size' : 6 ,
    'epochs' : 5 ,
    'callbacks' : None , # [ TensorBoard( log_dir='logs/{}'.format( time.time() ) ) ] ,
    'val_data' : None
}

recognizer.fit( [ X1 , X2 ], Y, hyperparameters=parameters)
recognizer.save_model('models/model.h5')

custom_images = recognizer.prepare_images_from_dir( 'custom_images/'  )
class_1_images = recognizer.prepare_images_from_dir( 'images/p1/' )
class_2_images = recognizer.prepare_images_from_dir( 'images/p2/' )

scores = list()
labels = list()
for image in custom_images:
    label = list()
    score = list()
    for sample in class_1_images :
        image , sample = image.reshape( ( 1 , -1 ) ) , sample.reshape((1 , -1 ) )
        score.append( recognizer.predict( [ image , sample ])[0] )
        label.append( 0 )
    for sample in class_2_images :
        image , sample = image.reshape( ( 1 , -1 ) ) , sample.reshape((1 , -1 ) )
        score.append( recognizer.predict( [ image , sample ])[0] )
        label.append( 1 )
    labels.append( label )
    scores.append( score )

scores = np.array( scores )
labels = np.array( labels )

for i in range( custom_images.shape[0] ) :
    index = np.argmax( scores[i] )
    label_ = labels[i][index]
    print( 'IMAGE {} is {} with confidence of {}'.format( i+1  , label_ , scores[i][index][0] ) )

