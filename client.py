import pickle
from flask import Flask, request, jsonify
from model_code.modeling import PrepTransformer
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd

app = Flask('app')

@app.route('/test',methods=['GET'])
def test():
	return 'Test ping to model app'

@app.route('/predict',methods=['GET','POST'])
def predict():
	with open('prep.bin','rb') as pin:
		prepTransformer = pickle.load(pin)
	with open('model.bin','rb') as modin:
		model = pickle.load(modin)

	arg_dict = request.args.to_dict(flat=True)
	for k, v in arg_dict.items():
		arg_dict[k]=int(v)
	print(f'arg_dict={arg_dict}')

	# compute prediction
	df = pd.DataFrame([arg_dict])
	df = prepTransformer.transform(df)
	prediction = model.predict(df)
	print(f'prediction={prediction}')

	result = {'mpg*':prediction.tolist()}
	return jsonify(result)

if __name__=='__main__':
	app.run(debug=True, host='0.0.0.0', port=9696)
