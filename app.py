from flask import Flask,render_template,request,session
from flask_session import Session
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
import cv2
import io
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow import keras
from tensorflow.python.keras.backend import set_session
import base64


nitrogen_deficiency_label = ['75%-100% nitrogen deficiency. use Urea and ammonium sulphate fertilizers', '30%-75% nitrogen deficiency. Use Ammonium phospate,Calcium amonium nitrate fertilizer', '1%-30% nitrogen deficiency.Use Phosphhrous and potassium fertilizers', 'No nitrogen deficiency']
nitrogen_deficiency_des = ['Leaf images with high Nitrogen deficiency(75%-100%)',
                            'Leaf images with medium Nitrogen deficiency(30%-75%)',
                            'Leaf images with low Nitrogen deficiency(1%-30%)',
                            'Leaf images with No Nitrogen deficiency']

nitrogen_deficiency_fer = ['Solution: Urea and Ammonium Sulphate fertilizers can be used.',
                           'Solution: Ammonium Phosphate and Calcium ammonium nitrate fertilizers can be used.',
                           'Solution: Phosphrous and Potassium fertilizers can be used.',
                           'Solution: Fertilizer is not required.']

disease_label = ['control', 'diseased']
# load CNN model
class SomeObj():
    def __init__(self):
        self.sess = tf.compat.v1.Session()
        self.graph = tf.compat.v1.get_default_graph()
        set_session(self.sess)
        self.model = keras.models.load_model('nitrogen_deficiency_image_model.h5')
    def sendM(self):
        return self.sess,self.model,self.graph

# load CNN model
class SomeObj1():
    def __init__(self):
        self.sess1 = tf.Session()
        self.graph1 = tf.get_default_graph()
        set_session(self.sess1)
        self.model1 = keras.models.load_model('Disease_Images_model.h5')
    def sendM1(self):
        return self.sess1,self.model1,self.graph1

global_obj = SomeObj()

global_obj1 = SomeObj1()




def load_data(image):
    X = []
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (255, 255)) /255
    X.append(image)
    X = np.array(X)
    return X




# create flask server
app = Flask(__name__)







@app.route('/index')
def index():
   return render_template('dashboard.html')

@app.route('/nitrogen_deficiency', methods = ['GET', 'POST'])
def nitrogen_deficiency():

    if request.method == 'POST':
    

        photo = request.files['file']
        in_memory_file = io.BytesIO()
        photo.save(in_memory_file)
        data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
        color_image_flag = 1
        img = cv2.imdecode(data, color_image_flag)
        

        retval, buffer = cv2.imencode('.jpg', img)
        jpg_as_text = base64.b64encode(buffer)
        data = str(jpg_as_text)
        data_image = str(jpg_as_text)

        img = load_data(img)

        sess,model,graph = global_obj.sendM()
        with graph.as_default():
            set_session(sess)
            data =  model.predict(img)
        print(type(data))
        
        data_r = nitrogen_deficiency_label[data.argmax()]
        data_d = nitrogen_deficiency_des[data.argmax()]
        data_f = nitrogen_deficiency_fer[data.argmax()]

        return render_template('result.html', class_data=data_r,data_d=data_d,data_f=data_f)
        
      

    return render_template('nitrogen_deficiency.html')


@app.route('/nitrogen_Disease', methods = ['GET', 'POST'])
def nitrogen_Disease():
   if request.method == 'POST':
    
        photo = request.files['file1']
        in_memory_file = io.BytesIO()
        photo.save(in_memory_file)
        data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
        color_image_flag = 1
        img = cv2.imdecode(data, color_image_flag)
        

        retval, buffer = cv2.imencode('.jpg', img)
        jpg_as_text = base64.b64encode(buffer)
        data = str(jpg_as_text)
        data_image = str(jpg_as_text)

        img = load_data(img)

        sess1,model1,graph1 = global_obj1.sendM1()
        with graph1.as_default():
            set_session(sess1)
            data =  model1.predict(img)
        print(data)

        data = disease_label[data.argmax()]




    
        return render_template('result.html', class_data=data,data_d = "",data_f="")
 

   return render_template('nitrogen_Disease.html')

@app.route('/result')
def result():
   return render_template('result.html')


if __name__ == '__main__':
    app.run(debug = True)
