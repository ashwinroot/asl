import requests
import json

'''
# After switching on the aws instance, the server must be
# switched on by running python3 run.py. Using the public IP,
# from the server to select the instane.
# Note : Public IP will change everytime when we restart an instance.
'''

URL = "http://34.236.171.100:5000/classify"

content_type = 'image/jpeg'
headers = {'content-type': content_type}



def post_image(img_file):
    """ post image and return the response """
    img = open(img_file, 'rb').read()
    response = requests.post(URL, data=img, headers=headers)
    return response
    
print(json.loads(post_image("asl-alphabet/asl_alphabet_test/A_test.jpg")))


