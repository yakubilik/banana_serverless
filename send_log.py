

def send_log(req):

    url= "http://192.168.0.148:8080"
    import requests
    import json

    payload = json.dumps(req)
    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)