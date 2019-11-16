from flask import Flask,request, render_template
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import pickle

app = Flask(__name__)

@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def prediction():
    model = pickle.load(open('model_1.pkl','rb'))
    val = np.array([float(request.form['Experience'])])
    P = PolynomialFeatures(degree=2)
    X = P.fit_transform(val.reshape(-1,1))
    output = model.predict(X)
    output = round(float(output),2)
    return 'Your salary prediction is '+ str(output)

if __name__=='__main__':
    app.run(debug = True, port='5001')