import numpy
from keras.datasets import cifar10
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
args = parser.parse_args()

print(args.mode)
np.random.seed(10)
(x_img_train,y_label_train),(x_img_test, y_label_test)=cifar10.load_data()
print('train:',len(x_img_train))
print('test :',len(x_img_test))
print('train_image :',x_img_train.shape)
print('train_label :',y_label_train.shape)
print('test_image :',x_img_test.shape)
print('test_label :',y_label_test.shape)

print(x_img_train[0][0][0]) #（50000，32，32，3）
x_img_train_normalize = x_img_train.astype('float32') / 255.0
x_img_test_normalize = x_img_test.astype('float32') / 255.0
print(x_img_train_normalize[0][0][0])


from keras.utils import np_utils
y_label_train_OneHot = np_utils.to_categorical(y_label_train)
y_label_test_OneHot = np_utils.to_categorical(y_label_test)
print(y_label_train_OneHot.shape)
print(y_label_train_OneHot[:5])

from keras.datasets import cifar10
import numpy as np
np.random.seed(10)
(x_img_train,y_label_train),(x_img_test,y_label_test)=cifar10.load_data()
print("train data:",'images:',x_img_train.shape,
      " labels:",y_label_train.shape)
print("test  data:",'images:',x_img_test.shape ,
      " labels:",y_label_test.shape)
x_img_train_normalize = x_img_train.astype('float32') / 255.0
x_img_test_normalize = x_img_test.astype('float32') / 255.0
from keras.utils import np_utils
y_label_train_OneHot = np_utils.to_categorical(y_label_train)
y_label_test_OneHot = np_utils.to_categorical(y_label_test)
y_label_test_OneHot.shape

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
model = Sequential()
model.add(Conv2D(filters=32,kernel_size=(3, 3),input_shape=(32, 32,3),
                 activation='relu', padding='same'))
model.add(Dropout(0.3))
model.add(Conv2D(filters=32, kernel_size=(3, 3),
                 activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64, kernel_size=(3, 3),
                 activation='relu', padding='same'))
model.add(Dropout(0.3))
model.add(Conv2D(filters=64, kernel_size=(3, 3),
                 activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=128, kernel_size=(3, 3),
                 activation='relu', padding='same'))
model.add(Dropout(0.3))
model.add(Conv2D(filters=128, kernel_size=(3, 3),
                 activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(2500, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1500, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation='softmax'))
print(model.summary())




if args.mode == 'test':
    model.load_weights('model.h5')
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    print(model.metrics_names)
    scores = model.evaluate(x_img_test_normalize,
                            y_label_test_OneHot, verbose=0)
    print(scores)
    print(scores[1])

else:
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    train_history=model.fit(x_img_train_normalize, y_label_train_OneHot,
                            validation_split=0.2,
                            epochs=20, batch_size=300, verbose=1)   


    model.save_weights('model.h5')



