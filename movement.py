import requests
from time import sleep

headers = {
    'Content-type': 'application/json',
    'X-Api-Key': '041AA6FA66184165A38B8D938C68A30E',
}


def move(x=0, y=0, z=10, home=False, speed=3000):
    json_data = {
        'command': f'G0 X{x} Y{y} Z{z} F1{speed}',
    }
    if home:
        json_data = {
            'command': f'G28'
        }
    print(requests.post(
        'http://10.1.25.244/api/printer/command', headers=headers, json=json_data))


move(home=True)
sleep(10)
move(y=240, z=50)
