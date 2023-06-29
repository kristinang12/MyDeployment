import pandas as pd
from flask import Flask, render_template, request
from waitress import serve
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

app = Flask(__name__)

@app.route('/linear_regression')
def linear_regression():
    return render_template('linear_regression.html')

@app.route('/linear_regression_results')
def linear_regression_results():
    return render_template('linear_regression_results.html')

@app.route('/linear_regression_ui', methods=['POST', 'GET'])
def linear_regression_ui():
    # Read the data file
    file_name = 'advertising_datasets.csv'  # Replace with your file name
    data = pd.read_csv(file_name)

    # Drop rows with NaN values
    data.dropna(axis=0, inplace=True)

    # Initializing the variables
    X = data[['Tv', 'Radio', 'Newspaper']]
    y = data['Sales'].values.reshape(-1, 1)

    # Splitting our dataset into Training and Testing dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fitting Linear Regression to the training set
    multiple_reg = LinearRegression()
    multiple_reg.fit(X_train, y_train)

    if request.method == 'POST':
        tv_budget = float(request.form['tv_budget'])
        radio_budget = float(request.form['radio_budget'])
        newspapers_budget = float(request.form['newspapers_budget'])

        # Make a prediction
        prediction = multiple_reg.predict([[tv_budget, radio_budget, newspapers_budget]])[0]

        return render_template('linear_regression_ui.html', prediction=prediction, tv_budget=tv_budget,
                               radio_budget=radio_budget, newspapers_budget=newspapers_budget)
    else:
        # Handle the GET request for rendering the form
        return render_template('linear_regression_ui.html')

@app.route('/')
def hello_world():
    return 'Hello, World!'

if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=8080)
