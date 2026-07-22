import streamlit as st
import numpy as np
import joblib
from PIL import Image
from datetime import datetime
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session
import os
import gdown
import joblib

file_id = '1JoopMIFsidHcin5CXUTjmPDTc8ekkuDJ'
# Destination path for the downloaded model
output = 'svm_model.pkl'

# Check if the file already exists to avoid redownloading
if not os.path.exists(output):
    # Download the file
    gdown.download(f'https://drive.google.com/uc?id={file_id}', output, quiet=False)

# Load the model with error handling
try:
    model = joblib.load(output)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None  # Set model to None if loading fails





# Function to preprocess the image before making predictions
def preprocess_image(image):
    img = Image.open(image)
    img = img.resize((512, 512))  
    img_array = np.array(img)
    img_normalized = img_array / 255.0  
    img_flattened = img_normalized.flatten()  
    return img_flattened


def fetch_ndvi_image(client_id, client_secret, coordinates, start_date, end_date):
    # Set up OAuth2 session
    client = BackendApplicationClient(client_id=client_id)
    oauth = OAuth2Session(client=client)
    token = oauth.fetch_token(
        token_url='https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token',
        client_secret=client_secret, include_client_id=True
    )

    evalscript = """
    //VERSION=3
    function setup() {
        return {
            input: [
                {
                    bands: ["B04", "B08"],
                },
            ],
            output: {
                id: "default",
                bands: 3,
            },
        }
    }

    function evaluatePixel(sample) {
        let ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04);

        if (ndvi < -0.941) return [0.47, 0.00, 0.00];
        else if (ndvi < -0.824) return [0.53, 0.00, 0.00];
        else if (ndvi < -0.706) return [0.6, 0.00, 0.00];
        else if (ndvi < -0.588) return [0.67, 0.00, 0.00];
        else if (ndvi < -0.471) return [0.73, 0.00, 0.00];
        else if (ndvi < -0.353) return [0.8, 0.00, 0.00];
        else if (ndvi < -0.235) return [0.87, 0.00, 0.00];
        else if (ndvi < -0.118) return [0.93, 0.00, 0.00];
        else if (ndvi < 0.0) return [1.0, 0.00, 0.00];
        else if (ndvi < 0.118) return [1.0, 0.00, 0.00];
        else if (ndvi < 0.235) return [1.0, 0.00, 0.00];
        else if (ndvi < 0.353) return [1.0, 0.8, 0.00];
        else if (ndvi < 0.471) return [1.0, 1.0, 0.00];
        else if (ndvi < 0.588) return [0.0, 1.0, 0.00];
        else if (ndvi < 0.706) return [0.0, 0.53, 0.00];
        else if (ndvi < 0.824) return [0.0, 0.4, 0.00];
        else return [0.0, 0.4, 0.00];
    }
    """

    request = {
        "input": {
            "bounds": {
                "properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coordinates],
                },
            },
            "data": [
                {
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "timeRange": {
                            "from": start_date,
                            "to": end_date,
                        }
                    },
                }
            ],
        },
        "output": {
            "width": 512,
            "height": 512,
            "responses": [
                {
                    "identifier": "default",
                    "format": {
                        "type": "image/jpeg",
                        "quality": 80,
                    },
                }
            ],
        },
        "evalscript": evalscript,
    }

    url = "https://sh.dataspace.copernicus.eu/api/v1/process"
    response = oauth.post(url, json=request)

    if response.status_code == 200:
        image_path = 'retrieved_ndvi_image.png'
        with open(image_path, 'wb') as f:
            f.write(response.content)
        return image_path
    else:
        st.error(f"Failed to retrieve image with status code: {response.status_code}")
        return None

# Set up the Streamlit app interface
st.title("Drought Image Classification App")
st.write("Upload an NDVI image or enter coordinates to retrieve an NDVI image for classification.")

# Option to choose between uploading an image or using coordinates
option = st.radio("Choose input method:", ('Upload Image', 'Use Coordinates'))

if option == 'Upload Image':
    uploaded_file = st.file_uploader("Choose an NDVI image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        processed_image = preprocess_image(uploaded_file).reshape(1, -1)
        prediction = model.predict(processed_image)
        labels = {0: 'No Drought', 1: 'Mild Drought', 2: 'Strong Drought'}
        st.write(f"Prediction: **{labels[prediction[0]]}**")

elif option == 'Use Coordinates':
    client_id = os.environ.get('COPERNICUS_CLIENT_ID', '')
    client_secret = os.environ.get('COPERNICUS_CLIENT_SECRET', '')
    coordinates_input = st.text_input("Enter coordinates (format: [[lon1, lat1], [lon2, lat2], ...]):")
    start_date = st.date_input("Start Date")
    end_date = st.date_input("End Date")

    # Format dates to ISO-8601 with timezone
    start_date_iso = datetime.combine(start_date, datetime.min.time()).isoformat() + 'Z'
    end_date_iso = datetime.combine(end_date, datetime.max.time()).isoformat() + 'Z'

    if st.button("Retrieve Image"):
        try:
            coordinates = eval(coordinates_input)  # Convert string to list
            image_path = fetch_ndvi_image(client_id, client_secret, coordinates, start_date_iso, end_date_iso)
            if image_path:
                st.image(image_path, caption='Retrieved NDVI Image', use_column_width=True)
                processed_image = preprocess_image(image_path).reshape(1, -1)
                prediction = model.predict(processed_image)
                labels = {0: 'No Drought', 1: 'Mild Drought', 2: 'Strong Drought'}
                st.write(f"Prediction: **{labels[prediction[0]]}**")
        except Exception as e:
            st.error(f"Error: {e}")
