from flask import Flask, render_template, request, redirect, jsonify
import json
import os
from flask_cors import CORS, cross_origin
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
cors = CORS(app)
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))  # Load the trained model
car = pd.read_csv('Cleaned data.csv')

@app.route('/get_names', methods=['GET'])

def get_names():
    company = request.args.get('company')
    if company:
        names = sorted(car[car['company'] == company]['name'].unique().tolist())
    else:
        names = []
    return jsonify(names)

@app.route('/', methods=['GET', 'POST'])
def index():
    if not os.path.isfile('Cleaned data.csv'):
        return 'Cleaned data.csv not found'
    companies = sorted(car['company'].unique())
    year = sorted(car['year'].unique(), reverse=True)
    fuel_type = car['fuel_type'].unique()

    companies.insert(0, 'Select Company')
    return render_template('index.html', companies=companies, years=year, fuel_types=fuel_type)



@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
  company = request.form.get('company', None)
  name = request.form.get('car_models', None)
  year = int(request.form.get('year', 0))
  kms_driven = int(request.form.get('kms_driven', 2))
  fuel_type = request.form.get('fuel_type', '')

  # Check for missing data
  if not all([company, name, year, kms_driven, fuel_type]):
      return jsonify({'error': 'Please fill in all required fields'}), 400

  # Get the column names from the car DataFrame
  columns = car.columns.tolist()

  # Create a DataFrame with the user input
  user_input = pd.DataFrame({
      columns[columns.index('name')]: [name],
      columns[columns.index('company')]: [company],
      columns[columns.index('year')]: [year],
      columns[columns.index('kms_driven')]: [kms_driven],
      columns[columns.index('fuel_type')]: [fuel_type]
  })

  # Make sure the column order matches the model's expected input
  user_input = user_input[columns]

  # Try-except block to handle potential model errors
  try:
      prediction = model.predict(user_input)
      predicted_price = np.round(prediction[0], 2)
      return jsonify({'predicted_price': predicted_price})
  except Exception as e:
      print(f"Error during prediction: {e}")
      return jsonify({'error': 'Internal server error'}), 500



if __name__=='__main__':
    app.run()