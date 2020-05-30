from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Conv2D,MaxPooling2D,ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.datasets import mnist
from keras.utils import np_utils
import keras

#load mnist dataset
(x_train,y_train),(x_test,y_test)=mnist.load_data()


# now storing the rows and column
img_rows =x_train[0].shape[0]
img_cols =x_train[1].shape[0]

#shaping the dateset for keras
x_train = x_train.reshape(x_train.shape[0],img_rows,img_cols,1)
x_test = x_test.reshape(x_test.shape[0],img_rows,img_cols,1)

#store the shape in single images
input_shape =(img_rows,img_cols,1)

#change image type to float
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#changing the range between 0 to 1
x_train /= 255
x_test /= 255
#encode output
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)


num_classes =y_test.shape[1]
num_pixels =x_train.shape[1] * x_train.shape[2]




#create model
model = Sequential()

#set of 'CRP' (Convolution,RELU,Pooling)
model.add(Conv2D(20,(5,5),
                    padding="same", 
                    input_shape =input_shape))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size= (2,2)))

model.add(Conv2D(50,(5,5),
                  padding="same"))

model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size= (2,2), strides = (2,2)))

#fully connected layers
model.add(Flatten())
model.add(Dense(500))
model.add(Activation("relu"))

#softmax for classification
model.add (Dense(num_classes))
model.add(Activation("softmax"))


model.compile(loss = 'categorical_crossentropy',
                optimizer=keras.optimizers.adadelta(),
                metrics =['accuracy'])

print(model.summary())

#training Parameters

batch_size = 128
epoch=2

history = model.fit (x_train,y_train,
                     batch_size=batch_size,
                     epochs=epoch,
                     validation_data=(x_test,y_test),
                     shuffle= True)

model.save("mnist_lenet.h5")

#evaluation
scores = model.evaluate(x_test,y_test,verbose=1)
print('test loss:',scores[0])
print('test accuracy:',scores[1])
try:
    f = open('/project/out.txt','w')
    f.write(str(int(scores[1]*100)))
except:
    print(end="")
finally:
    f.close()
