#import necessary modules
#following program will automatically find best model and using that best model it will predict the price of house
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, r2_score, mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeRegressor
from flask import Flask, render_template, request, jsonify
import pickle

def test_model(model, X_train, y_train):
    cv = KFold(n_splits=3, shuffle=True, random_state=45)
    r2 = make_scorer(r2_score)
    r2_val_score = cross_val_score(model, X_train, y_train, cv=cv, scoring=r2)
    return r2_val_score.mean()


def saveModel():
    # Load data
    df = pd.read_csv("house_prices.csv")
    df.dropna(inplace=True)

    # Convert categorical variables to numerical
    df['Brick'] = pd.factorize(df['Brick'])[0]
    df['Neighborhood'] = pd.factorize(df['Neighborhood'])[0]

    # Prepare data
    X = df.drop(['Home','Price', 'Offers'], axis=1).values
    y = np.array(df["Price"])

    # Initialize variables to keep track of the best model
    best_model = None
    best_score = float('-inf')  # Initialize with negative infinity
    best_model_name = ""

    # Models to try
    models = [LinearRegression(), DecisionTreeRegressor(), RandomForestRegressor()]
    fittedModel = None

    # Iterate over models
    for model in models:
    
        score = test_model(model, X, y)
        # Update best model if current model is better
        if score > best_score:
            best_model = model
            best_score = score
            fittedModel = model.fit(X, y)
            best_model_name = model.__class__.__name__

    # Save best model
    with open("best_model.pkl", "wb") as f:
        pickle.dump(fittedModel, f)
    # Output best model's accuracy on test data
    #print(f"Best model: {best_model_name} with accuracy of {best_score:.2f}")

def predict(data):
    # Convert received data into a DataFrame
    df_test = pd.DataFrame(data, index=[0])
    print(df_test)
    df_test['Brick'] = pd.factorize(df_test['Brick'])[0]
    df_test['Neighborhood'] = pd.factorize(df_test['Neighborhood'])[0]
    test_features = np.array(df_test)

    # Predict using best model
    with open("best_model.pkl", "rb") as f:
        fittedModel = pickle.load(f)
        predictions = fittedModel.predict(test_features)
        # return the predicted price
        return int(predictions[0])

app = Flask(__name__)
  
@app.route('/predict', methods=['POST'])
def prediction():
    #check whether model exists, if not then create model
    isModelExist = False
    try:
        with open("best_model.pkl", "rb") as f:
            isModelExist = True
    except:
        isModelExist = False
    
    if not isModelExist:
        saveModel()

    # Retrieve data from the request
    # Retrieve form data from the request
    sqft = int(request.form['SqFt'])
    bedrooms = int(request.form['Bedrooms'])
    bathrooms = int(request.form['Bathrooms'])
    brick = request.form['Brick']
    neighborhood = request.form['Neighborhood']
    
    # Prepare data as a dictionary
    data = {
        'SqFt': sqft,
        'Bedrooms': bedrooms,
        'Bathrooms': bathrooms,
        'Brick': brick,
        'Neighborhood': neighborhood
    }
    print("suri",data)
    # Call the predict function with the received data
    predicted_price = predict(data)
    # Return the predicted price as JSON response
    return jsonify({'predicted_price': predicted_price})

@app.route('/')
def index():
    return render_template("index.html")

if __name__ == '__main__':
    app.debug = True
    app.run()
