import requests
import json
import time

API_KEY =  '81195cd8-498d-4bff-95df-d4fa82ecddd6'
LOG_ID = 'c6c20374-dd3f-4e21-85eb-92511d44a767'

def handle_response(resp):
    response = resp
    time.sleep(1)
    if response.status_code == 200:
        print(response.content)
    else:
        print(response.status_code)

def make_request(provided_url=None):
    headers = {'x-api-key': API_KEY}

    url = "https://US.rest.logs.insight.rapid7.com/download/logs/c6c20374-dd3f-4e21-85eb-92511d44a767"
    params = {"time_range": "last 10 mins"}
    req = requests.get(url, headers=headers, params=params)
    return req

def get_log():
    req = make_request()
    handle_response(req)

def start():
    get_log()

if __name__ == '__main__':
    start()