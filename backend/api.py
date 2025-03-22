from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import lightgbm as lgb
import unicodedata
import os

app = Flask(__name__)
CORS(app)

# Mapping property types to numerical values for standardization purposes.
# The arabic and english.
property_type_mapping = {
    'apartment': 0.0,
    'شقة': 0.0,
    'villa': 0.99,
    'فيلا': 0.99,
    'floor': 0.49,
    'دور': 0.49
}

# Function to normalize text for consistent processing.
# Converts the text to Unicode Normal Form (NFKC) to handle text variations.
# Strips leading/trailing whitespace and converts the text to lowercase.
def normalize_text(text):
    if isinstance(text, str):  # Check if the input is a string
        return unicodedata.normalize('NFKC', text).strip().lower()
    return text  # Return non-string inputs unchanged

# Mapping English city names to their Arabic equivalents.
english_to_arabic_city_map = {
    'riyadh': 'الرياض',
    'jeddah': 'جدة',
    'dammam': 'الدمام',
    'khobar': 'الخبر'
}

train_data = pd.read_csv('./backend/train_90.csv')
befpreprocess_data = pd.read_csv('./backend/befpreprocess.csv')

befpreprocess_data['city'] = befpreprocess_data['city'].apply(normalize_text)
befpreprocess_data['district'] = befpreprocess_data['district'].apply(normalize_text)

# These helps in light gbm model is normalized.
# Creating a mapping of city names to normalized numerical values.
city_mapping = {city: idx / len(befpreprocess_data['city'].unique()) for idx, city in enumerate(befpreprocess_data['city'].unique())}
# Creating a mapping of district names to normalized numerical values.
district_mapping = {district: idx / len(befpreprocess_data['district'].unique()) for idx, district in enumerate(befpreprocess_data['district'].unique())}

X_train = train_data.drop(columns=['price'])
y_train = train_data['price']
lgb_train = lgb.Dataset(X_train, y_train)

params = {
    'objective': 'regression',
    'metric': ['mae', 'rmse'],
    'boosting_type': 'gbdt',
    'learning_rate': 0.1,
    'force_row_wise': True,
    'max_bin': 550,
    'subsample_for_bin': 200000,
    'min_child_samples': 10,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
}

model = lgb.train(params, lgb_train, num_boost_round=200)

@app.route('/districts', methods=['GET'])
# Function to retrieve and process districts based on a given city name.
def get_districts():
    # Retrieve the 'city' parameter from the request and normalize it for consistency.
    city_name = normalize_text(request.args.get('city', ''))

    # Check if the provided city name exists in the dataset.
    if city_name not in befpreprocess_data['city'].unique():
        # Log an error message if the city is not found in the dataset.
        print(f"Received city: '{city_name}' not found in city data.")
        # Return a JSON response with an error message and a 400 status code.
        return jsonify({'error': 'تم اختيار مدينة غير صحيحة.'}), 400

    # Filter the dataset to get districts corresponding to the specified city.
    districts = list(set(befpreprocess_data[befpreprocess_data['city'] == city_name]['district'].tolist()))
    # Normalize the district names for consistency.
    districts = [normalize_text(district) for district in districts]

    # Clean district names by removing the prefix "حي " and extra spaces.
    cleaned_districts = [district.replace("حي ", "").strip() for district in districts]

    print(f"Available districts for '{city_name}': {cleaned_districts}")
    for district in cleaned_districts:
        encoded_value = district_mapping.get(district, 'N/A')
        print(f"District: '{district}' -> Encoded Value: {encoded_value}")

    # Return a JSON response containing the cleaned list of districts.
    return jsonify({'districts': cleaned_districts})

def preprocess_input(data):
    city = normalize_text(data.get('city', ''))
    district = normalize_text(data.get('district', ''))
    property_type = normalize_text(data.get('property_type', ''))

    print(f"Preprocessing input: city='{city}', district='{district}', property_type='{property_type}'")

    # Print the encoded mappings before checking
    print("City Mapping:", city_mapping)
    print("District Mapping:", district_mapping)
    print("Property Type Mapping:", property_type_mapping)

    if city not in city_mapping:
        print(f"City '{city}' not found in city_mapping.")
        return "Invalid city value.", None
    if district not in district_mapping:
        print(f"District '{district}' not found in district_mapping.")
        return "Invalid district value.", None
    if property_type not in property_type_mapping:
        print(f"Property type '{property_type}' not found in property_type_mapping.")
        return "Invalid property type value. Choose from: شقة, فيلا, دور", None

    try:
        city_encoded = city_mapping[city]
        district_encoded = district_mapping[district]
        property_type_encoded = property_type_mapping[property_type]
        area = float(data['area'])
        rooms = int(data['rooms'])
        bathrooms = int(data['bathrooms'])

        print(f"Encoded Values: city='{city}' -> {city_encoded}, district='{district}' -> {district_encoded}, property_type='{property_type}' -> {property_type_encoded}")
    except ValueError as e:
        print(f"Error in input conversion: {e}")
        return "Invalid input format. Ensure numeric values for area, rooms, and bathrooms.", None

    input_data = pd.DataFrame([{
        'city': city_encoded,
        'district': district_encoded,
        'property_type': property_type_encoded,
        'area': area,
        'rooms': rooms,
        'bathrooms': bathrooms,
    }])

    print(f"Encoded input data:\n{input_data}")
    return None, input_data

@app.route('/predict', methods=['POST'])
# Function to handle price prediction requests.
def predict():
    # Retrieve JSON data sent in the request body.
    data = request.get_json()

    # Preprocess the input data to ensure it is in the correct format for the model.
    error, input_data = preprocess_input(data)

    # If there is an error in preprocessing, return an error response with a 400 status code.
    if error:
        return jsonify({'error': error}), 400

    print("Making prediction with input data:")
    print(input_data)

    # Use the trained model to make a prediction based on the input data.
    prediction = model.predict(input_data)
    # Extract the predicted price from the model's output and round it to 2 decimal places.
    predicted_price = round(prediction[0], 2)

    print(f"Predicted price: {predicted_price}")

    # Return the predicted price as a JSON response.
    return jsonify({'predicted_price': predicted_price})

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=10000)


