import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from flask import *
from werkzeug.utils import secure_filename

##Define the flask app
app = Flask(__name__)

##Load trained model
model = load_model("model.h5")

def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size = (64, 64))

    #Preprocessing the image
    x = image.img_to_array(img)
    ##Scaling
    x = x/255.0
    x = np.expand_dims(x, axis = 0)

    preds = model.predict(x)
    preds = np.argmax(preds)

    if preds == 0:
        preds = "Great!!!, You don't have Brain Tumor Disease."
    else:
        preds = "Sorry, You have Brain Tumor Disease, kindly contact your doctor."

    return preds

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predict', methods = ['GET', 'POST'])
def upload():
    if request.method == 'POST':
        ## Get the file from post request
        f = request.files['file']
        ##save the file './uploads'
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        ##Make predictions
        preds = model_predict(file_path, model)
        result = preds
        return result

    return None

if __name__ == '__main__':
    app.run(port = 5001, debug= True)





