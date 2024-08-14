import pandas as pd
import pickle
from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)

def load_locations_data(file_path):
    df = pd.read_csv(file_path)
    df = df.drop(columns=['Unnamed: 0'], errors='ignore') 
    
    locations_data = df.groupby('city')['location'].apply(lambda x: sorted(set(x))).to_dict()
    
    
    return locations_data


model_path = "House Prediction\\House Price Model.pkl"
ct_path = "House Prediction\\column_transformer.pkl"

with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

with open(ct_path, 'rb') as ct_file:
    ct = pickle.load(ct_file)

locations_data = load_locations_data("House Prediction\\Cleaned_data_for_model.csv")

@app.route('/', methods=['GET', 'POST'])
def index():
    selected_city = request.form.get('city')
    locations = locations_data.get(selected_city, []) if selected_city else []

    if request.method == 'POST' and 'predict' in request.form:
        try:
            property_type = request.form.get('property_type')
            location = request.form.get('location')
            city = request.form.get('city')
            baths = request.form.get('baths')
            purpose = request.form.get('purpose')
            bedrooms = request.form.get('bedrooms')
            marla = request.form.get('marla') 

            if None in [city, location, marla, baths, purpose, property_type, bedrooms]:
                raise ValueError("One or more required fields are missing or invalid.")

            marla = float(marla)
            baths = int(baths)
            bedrooms = int(bedrooms)

            data = pd.DataFrame([{
                'property_type': property_type,
                'location': location,
                'city': city,
                'baths': baths,
                'purpose': purpose,
                'bedrooms': bedrooms,
                'Area_in_Marla': marla,
            }])

            #print(f"Columns in input data: {data.columns.tolist()}")

            try:
                data_transformed = ct.transform(data)
            except Exception as e:
                print(f"Error during transformation: {e}")
                return render_template('index.html', 
                                       cities=locations_data.keys(), 
                                       locations=locations, 
                                       selected_city=selected_city,
                                       error="Error during data transformation.")

            try:
                prediction = model.predict(data_transformed)
                prediction = np.exp(prediction)
            except Exception as e:
                print(f"Error during prediction: {e}")
                return render_template('index.html', 
                                       cities=locations_data.keys(), 
                                       locations=locations, 
                                       selected_city=selected_city,
                                       error="Error during prediction.")
            
            return render_template('index.html', 
                                   cities=locations_data.keys(), 
                                   locations=locations, 
                                   selected_city=selected_city,
                                   prediction=prediction[0])
        
        except ValueError as e:
            return render_template('index.html', 
                                   cities=locations_data.keys(), 
                                   locations=locations, 
                                   selected_city=selected_city,
                                   error=str(e))
    
    return render_template('index.html', 
                           cities=locations_data.keys(), 
                           locations=locations, 
                           selected_city=selected_city)

@app.route('/update_locations', methods=['POST'])
def update_locations():
    selected_city = request.form.get('city')
    locations = locations_data.get(selected_city, []) if selected_city else []
    return {
        'locations': locations
    }


if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=5000)