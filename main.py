from flask import Flask, jsonify, request
from gathering import pipeline_gathering
from model_ import prepare_data, predict_model, train_model

app = Flask(__name__)

DATA = None
MODEL = None
TRAIN = None
TEST = None
SCALER = None

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/load')
def load_data():
    global DATA
    DATA = pipeline_gathering()
    return 'Finish!'

@app.route('/data')
def get_data():
    global DATA
    return jsonify(DATA), 200

@app.route('/train')
def train():
    global DATA
    global MODEL
    global TRAIN
    global TEST
    global SCALER

    if DATA is None:
        return 'Please, load data with /load'
    
    MODEL, TRAIN, TEST, SCALER = train_model(DATA)

    return 'Training Complete!'


@app.route('/predict')
def predicted():
    global DATA
    global MODEL
    global TRAIN
    global TEST
    global SCALER

    days = request.args.get('days', default=30, type=int)

    if TRAIN is None or MODEL is None or TEST is None or SCALER is None:
        return 'Please, execute /train before.'

    #data_input = request.get_json()
    predict_data = predict_model(MODEL, SCALER, TRAIN, days)

    return jsonify(predict_data)


    
if __name__=="__main__":
    print(('* loading AVAAIL ML API starting server'))
    #global DATA
    load_data()
    app.run(host='0.0.0.0')