from PIL import Image
import matplotlib.pyplot as plt
import requests
from io import BytesIO

def fetch_img(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            return img
        else:
            print(f"Failed to retrieve image. Status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None



