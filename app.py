import pickle
from flask import Flask,request,app, jsonify, render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
## Load the model from the pickle file made in the notebook. 
regmodel=pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template("home.html")
# this is the home route to load home.html 

@app.route('/predict', methods=['POST'])
def predict():
    input=[float(x) for x in request.form.values()]
    # gets the inputs from the submitted form's values. 
    print(input)
    input_array = np.array(input).reshape(1, -1)
    # reshapes the array of the inputs into a format the model can understand. 
    output=regmodel.predict(input_array)[0]
    # makes the prediction of the inputs passed in with the model. 
    return render_template("home.html", prediction_test=output)
# returns the home.html template with prediction_test value passed in so it displays on the page. 

if __name__=="__main__":
    app.run(debug=True, host='0.0.0.0')