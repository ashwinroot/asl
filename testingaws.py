import requests
import json

URL = "http://34.236.171.100:5000/classify"

content_type = 'image/jpeg'
headers = {'content-type': content_type}



def post_image(img_file):
    """ post image and return the response """
    img = open(img_file, 'rb').read()
    response = requests.post(URL, data=img, headers=headers)
    return response

post_image("asl-alphabet/asl_alphabet_test/A_test.jpg")


