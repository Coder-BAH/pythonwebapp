import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

model=pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
      return render_template('index.html')
    
    
@app.route('/predict',methods=['POST'])
def predict():
    data1=request.form['a']
    data2=request.form['b']
    data3=request.form['c']
    data4=request.form['d']
    data5=request.form['e']
    data6=request.form['f']
    data7=request.form['g']
    data8=request.form['h']
    data9=request.form['i']
    data10=request.form['j']
    data11=request.form['j']
    
    arr=np.array([[data1,data2,data3,data4,data5,data6,data7,data8,data9,data10,data11]])
    
    
    pred_score=model.predict(arr)
    output =prediction[0]/10
    return render_template('index.html', prediction_text='The IMDb Score is {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
