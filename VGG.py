import os
from cv2 import imread
import matplotlib.pyplot as plt
from sklearn import metrics
import keras
from keras.layers import Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.applications import VGG19 #resnet 50
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")

path = "brain_tumor_dataset"

image_generator = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=2,
    vertical_flip=2,
    validation_split=0.1)

training_set = image_generator.flow_from_directory(batch_size=16,
                                                 directory=path,
                                                 shuffle=True,
                                                 target_size=(224,224),
                                                 subset="training",
                                                 color_mode='rgb',
                                                 class_mode='binary')

testing_set = image_generator.flow_from_directory(batch_size=16,
                                                 directory=path,
                                                 shuffle=True,
                                                 target_size=(224,224),
                                                 subset="validation",
                                                 color_mode='rgb',
                                                 class_mode='binary')

yes_path = path+"/yes/"+os.listdir(path+"/yes/")[4]
tumour = imread(yes_path)
#plt.imshow(tumour)


no_path = path+"/no/"+os.listdir(path+"/no/")[4]
tumour = imread(yes_path)
#plt.imshow(tumour)

testing_set.class_indices

model = VGG19(
      input_shape = (224,224,3),
      include_top = False,
      weights = 'imagenet'
    )

for layers in model.layers:
    layers.trainable = False

x = Flatten()(model.output)
x = Dropout(0.4)(x)
x = Dense(1, activation = "sigmoid")(x)

model = keras.Model(model.input, x)
model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = "accuracy")
model.summary()

hist = model.fit_generator(training_set,validation_data = testing_set, epochs =20)

hist = hist.history
plt.plot(hist["accuracy"])
plt.plot(hist["val_accuracy"])
plt.title("Accuracy plot")
plt.legend(["train","test"])
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.show()
plt.plot(hist["loss"])
plt.plot(hist["val_loss"])
plt.title("Accuracy loss")
plt.legend(["train","test"])
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

model.evaluate(testing_set)
ypred = model.predict(testing_set[0][0])
ypred = np.array([1 if x > 0.5 else 0 for x in ypred])
#ypred = ypred.astype(int)
ytest = testing_set[0][-1]

print('Confusion_Matrix:\n',confusion_matrix(ytest, ypred))
print('VGG19_CNN:\n',classification_report(ypred,ytest))


#define metrics
y_pred = model.predict(testing_set[0][0])
fpr, tpr, _ = metrics.roc_curve(ytest,  y_pred)

#create ROC curve
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
