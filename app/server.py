
from flask import Flask, render_template, request
import json
from flask_cors import CORS
import numpy as np
from math import floor

app = Flask(__name__)
CORS(app)

PHONES_INFO = {"Apple":['Iphone 4', 'Iphone 4s', 'Iphone 5', 'Iphone 5s', 'Iphone 6'], "Motorola":['G1', 'G2', 'G3'], "OnePlus":['One','Two', 'X'],
"Samsung":['Galaxy y', 'Galaxy Win', 'Galaxy S2','Grand 2', 'Galaxy Ace']
, "Xiaomi":['Redmi 1s','Redmi 2','Redmi Note 3']}
COMPANY_RATING = {'Apple':5,'Motorola':3,'OnePlus':4,'Samsung':3,'Xiaomi':2}
ISSUE_RATING = {'Hang':1,'None':0,'Battery':2,'Hang+Battery':3,'Microphones':2.5,'Battery+Microphones':4,'Hang+Microphones':3.5,'Hang+Wifi':4.5,'Wifi+Microphones':5}
MODEL_RATING ={'Iphone 4':2,'Iphone 4s':3,'Iphone 5':3.5,'Iphone 5s':4,'Iphone 6':5,
                'G1':2,'G2':3,'G3':4,
                'One':3.5,'Two':4,'X':3,
                'Redmi 1s':2,'Redmi 2':3,'Redmi Note 3':5,
                'Galaxy y':2, 'Galaxy Win':3, 'Galaxy S2':5,'Grand 2':4, 'Galaxy Ace':2,'Galaxy Y':2}

MODEL_FILE = '../models/model1'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/plot.html')
def plot():
    return render_template('plot.html')


@app.route('/get_models/<company>')
def get_models(company):
    print "here "+company
    return json.dumps(PHONES_INFO[company])


@app.route('/predict', methods=['GET','POST'])
def predict():
    data = dict(request.form)
    company = data['company'][0]
    issue = data['issue'][0]
    phoneModel = data['model'][0]
    boughtAt = int(data['boughtAt'][0])
    # company = 'Apple'
    # issue = 'Battery'
    # phoneModel = 'Iphone 6'
    # boughtAt = 12345
    theta = []
    with open(MODEL_FILE,'r') as f:
        model = json.loads(f.read())
        theta = model['theta']
    meanIn,stdDevIn = model['input_scaling_factors'][0],model['input_scaling_factors'][1]
    overallRating = int(COMPANY_RATING[company]) * int(MODEL_RATING[phoneModel])
    issueRating = ISSUE_RATING[issue]
    months = [3,6,9,12,15,18,24]
    passingX = []
    for x in months:
        temp=[1]
        scaledFeature = (overallRating-meanIn[0])/stdDevIn[0]
        temp.append(scaledFeature)
        scaledFeature = (boughtAt-meanIn[1])/stdDevIn[1]
        temp.append(scaledFeature)
        scaledFeature = (x-meanIn[2])/stdDevIn[2]
        temp.append(scaledFeature)
        scaledFeature = (issueRating-meanIn[3])/stdDevIn[3]
        temp.append(scaledFeature)
        passingX.append(temp)
    print passingX
    predictedY = np.dot(passingX,theta)
    scaledY = []
    meanOut = model['output_scaling_factors'][0]
    stdDevOut = model['output_scaling_factors'][1]
    for x in predictedY:
        scaledY.append(floor(x*stdDevOut+meanOut))
    print scaledY
    dataList={}
    dataList['company']=company
    dataList['phoneModel']=phoneModel
    dataList['issue']=issue
    dataList['predictedY'] = scaledY
    return render_template('predictedValues.html',dataList=dataList)

@app.route('/plotError')
def plotError():
    send = {}
    J = []
    with open(MODEL_FILE,'r') as f:
        model = json.loads(f.read())
        J = model['J']
        if(len(J)>=5000):
            J = J[:5000]
        else:
            print "error"
    send["J"] = J
    return json.dumps(send)

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)
