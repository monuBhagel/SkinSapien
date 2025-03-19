import requests

url = "http://127.0.0.1:5000/predict"
image_path = "/Users/rajeshkumar/skinsapiens-webapp/skin-disease-dataset/test_set/BA- cellulitis/BA- cellulitis (63).jpg"  # Replace with your image path

files = {"file": open(image_path, "rb")}
response = requests.post(url, files=files)

print(response.json()) 