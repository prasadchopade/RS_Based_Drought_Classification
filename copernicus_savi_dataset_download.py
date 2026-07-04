from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session
from datetime import datetime, timedelta
import os

# Your client credentials
client_id = ''
client_secret = ''

# Create a session
client = BackendApplicationClient(client_id=client_id)
oauth = OAuth2Session(client=client)

# Get token for the session
token = oauth.fetch_token(
    token_url='https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token',
    client_secret=client_secret,
    include_client_id=True
)

# Define the polygon coordinates for Andhra Pradesh region
polygon_coordinates = [
  [71.0, 26.5], [71.0, 26.9], [71.5, 26.9], [71.5, 26.5], [71.0, 26.5]
]

# Create directory for Andhra Pradesh
output_dir = "rajasthan_savi"
os.makedirs(output_dir, exist_ok=True)

# SAVI evalscript (keeping it unchanged as requested)
evalscript = """
//VERSION=3

// Soil Adjusted Vegetation Index  (abbrv. SAVI)
// General formula: (800nm - 670nm) / (800nm + 670nm + L) * (1 + L)

function setup() {
    return {
        input: ["B03", "B04", "B08", "dataMask"],
        output: [
            { id: "default", bands: 4 },
            { id: "index", bands: 1, sampleType: "FLOAT32" },
            { id: "eobrowserStats", bands: 2, sampleType: 'FLOAT32' },
            { id: "dataMask", bands: 1 }
        ]
    };
}

let L = 0.428; // L = soil brightness correction factor could range from (0 - 1)

const ramp = [
    [-0.5, 0x0c0c0c],
    [-0.2, 0xbfbfbf],
    [-0.1, 0xdbdbdb],
    [0, 0xeaeaea],
    [0.025, 0xfff9cc],
    [0.05, 0xede8b5],
    [0.075, 0xddd89b],
    [0.1, 0xccc682],
    [0.125, 0xbcb76b],
    [0.15, 0xafc160],
    [0.175, 0xa3cc59],
    [0.2, 0x91bf51],
    [0.25, 0x7fb247],
    [0.3, 0x70a33f],
    [0.35, 0x609635],
    [0.4, 0x4f892d],
    [0.45, 0x3f7c23],
    [0.5, 0x306d1c],
    [0.55, 0x216011],
    [0.6, 0x0f540a],
    [1, 0x004400],
];

const visualizer = new ColorRampVisualizer(ramp);

function evaluatePixel(samples) {
    const val = (samples.B08 - samples.B04) / (samples.B08 + samples.B04 + L) * (1.0 + L);
    const indexVal = samples.dataMask === 1 ? val : NaN;
    const imgVals = visualizer.process(val);

    return {
        default: imgVals.concat(samples.dataMask),
        index: [indexVal],
        eobrowserStats: [val, isCloud(samples) ? 1 : 0],
        dataMask: [samples.dataMask]
    };
}

function isCloud(samples) {
    const NGDR = index(samples.B03, samples.B04);
    const bRatio = (samples.B03 - 0.175) / (0.39 - 0.175);
    return bRatio > 1 || (bRatio > 0 && NGDR > 0);
}
"""

# Loop through each month from 2014 to 2023
start_date = datetime(2018, 1, 1)
end_date = datetime(2023, 12, 31)

current_date = start_date
while current_date <= end_date:
    # Define the time range for the current month
    next_month = current_date + timedelta(days=30)
    time_range_from = current_date.strftime('%Y-%m-%dT%H:%M:%SZ')
    time_range_to = next_month.strftime('%Y-%m-%dT%H:%M:%SZ')

    # Define the request payload for the current month
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
    url = "https://sh.dataspace.copernicus.eu/api/v1/process"
    response = oauth.post(url, json=request)

    # Check if the response is binary (e.g., for image files)
    if response.status_code == 200:
        # Save the image file locally with a unique name for each month
        file_name = f'{output_dir}/rajasthan_savi_{current_date.strftime("%Y_%m")}.png'
        with open(file_name, 'wb') as f:
            f.write(response.content)
        print(f"Image for {current_date.strftime('%B %Y')} saved successfully.")
    else:
        print(f"Failed for {current_date.strftime('%B %Y')} with status code: {response.status_code}")

    # Move to the next month
    current_date = next_month
