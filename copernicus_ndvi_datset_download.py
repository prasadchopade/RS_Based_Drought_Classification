# pip install oauth
# pip install requests requests_oauthlib

from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session
from datetime import datetime, timedelta

# Your client credentials
client_id = ''
client_secret = ''

# Create a session
client = BackendApplicationClient(client_id=client_id)
oauth = OAuth2Session(client=client)

# Get token for the session
token = oauth.fetch_token(token_url='https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token',
                          client_secret=client_secret, include_client_id=True)

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
  let ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04)

  if (ndvi < -0.5) return [0.05, 0.05, 0.05]
  else if (ndvi < -0.2) return [0.75, 0.75, 0.75]
  else if (ndvi < -0.1) return [0.86, 0.86, 0.86]
  else if (ndvi < 0) return [0.92, 0.92, 0.92]
  else if (ndvi < 0.025) return [1, 0.98, 0.8]
  else if (ndvi < 0.05) return [0.93, 0.91, 0.71]
  else if (ndvi < 0.075) return [0.87, 0.85, 0.61]
  else if (ndvi < 0.1) return [0.8, 0.78, 0.51]
  else if (ndvi < 0.125) return [0.74, 0.72, 0.42]
  else if (ndvi < 0.15) return [0.69, 0.76, 0.38]
  else if (ndvi < 0.175) return [0.64, 0.8, 0.35]
  else if (ndvi < 0.2) return [0.57, 0.75, 0.32]
  else if (ndvi < 0.25) return [0.5, 0.7, 0.28]
  else if (ndvi < 0.3) return [0.44, 0.64, 0.25]
  else if (ndvi < 0.35) return [0.38, 0.59, 0.21]
  else if (ndvi < 0.4) return [0.31, 0.54, 0.18]
  else if (ndvi < 0.45) return [0.25, 0.49, 0.14]
  else if (ndvi < 0.5) return [0.19, 0.43, 0.11]
  else if (ndvi < 0.55) return [0.13, 0.38, 0.07]
  else if (ndvi < 0.6) return [0.06, 0.33, 0.04]
  else return [0, 0.27, 0]
}
"""

# Define polygons for drought-prone regions
regions = {
    'Rajasthan': [[70.0, 25.0], [70.0, 28.0], [73.0, 28.0], [73.0, 25.0], [70.0, 25.0]],
    'Maharashtra': [[77.0, 19.0], [77.0, 22.0], [80.0, 22.0], [80.0, 19.0], [77.0, 19.0]],
    'Karnataka': [[75.0, 14.0], [75.0, 17.0], [78.0, 17.0], [78.0, 14.0], [75.0, 14.0]],
    'Telangana': [[78.0, 16.0], [78.0, 19.0], [81.0, 19.0], [81.0, 16.0], [78.0, 16.0]]
}

# Loop through regions and process each one
for region_name, polygon_coordinates in regions.items():
    current_date = datetime(2018, 1, 1)
    end_date = datetime(2023, 12, 31)
    while current_date <= end_date:
        next_month = current_date + timedelta(days=30)
        time_range_from = current_date.strftime('%Y-%m-%dT%H:%M:%SZ')
        time_range_to = next_month.strftime('%Y-%m-%dT%H:%M:%SZ')

        # Define the request payload for the current month and region
        request = {
            "input": {
                "bounds": {
                    "properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [polygon_coordinates],
                    },
                },
                "data": [
                    {
                        "type": "sentinel-2-l2a",
                        "dataFilter": {
                            "timeRange": {
                                "from": time_range_from,
                                "to": time_range_to,
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

        # Send the request to the Sentinel Hub API
        response = oauth.post(url, json=request)

        # Check if the response is successful and save the image
        if response.status_code == 200:
            file_name = f'{region_name}_{current_date.strftime("%Y_%m")}.png'
            with open(file_name, 'wb') as f:
                f.write(response.content)
            print(f"Image for {region_name} ({current_date.strftime('%B %Y')}) saved successfully.")
        else:
            print(f"Failed for {region_name} ({current_date.strftime('%B %Y')}) with status code: {response.status_code}")

        # Move to the next month
        current_date = next_month
