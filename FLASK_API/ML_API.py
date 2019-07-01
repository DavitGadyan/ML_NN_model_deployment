import numpy as np
from sklearn.externals import joblib
from flask import Flask, abort, jsonify, request

log_reg=joblib.load(r'C:\Users\Gaya\Desktop\log_reg1.pkl')

app=Flask(__name__)

@app.route('/api',methods=['POST'])
def make_predict():
	data=request.get_json(force=True)

	predict_request=[data['col1'],data['col2'],data['col3'],data['col4']]
	predict_request=np.array([predict_request])
	y_pred=log_reg.predict(predict_request)

	output= [y_pred[0]]

	return jsonify(results=int(output[0]))


if __name__=='__main__':
	app.run(port=9000,debug=True)