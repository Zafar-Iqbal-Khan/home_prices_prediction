import joblib
from flask import Flask,render_template,request
import numpy as np
import json



__locations=None
__data_columns=None
 

model=joblib.load('banglore_home_prices_model.pkl')


with open("columns.json",'r') as f:
        __data_columns=json.load(f)['data_columns']
        __locations=__data_columns[3:]

def get_estimated_price(location,sqft,bhk,bath,):
    try:
        loc_index=__data_columns.index(location.lower())
    except:
        loc_index=-1

    x=np.zeros(len(__data_columns))
    x[0]=sqft
    x[1]=bath
    x[2]=bhk

    if(loc_index>=0):
        x.loc_index=1
    
    return round(model.predict([x])[0],2)


#print(get_estimated_price('devarabeesana halli',1000,3,3))


app=Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():  
    input_features = [x for x in request.form.values()]
    area = input_features[0]
    bhk = input_features[1]
    bath = input_features[2]
    loacation = input_features[3]
    
    predicted_price = get_estimated_price(loacation,area,bhk,bath) 
    return render_template('index.html', prediction_text='Predicted Price of Bangalore House is {}'.format(predicted_price))
    
    
if __name__=='__main__':
    app.run(debug=True)