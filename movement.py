import requests
from time import sleep

headers = {
    'Content-type': 'application/json',
    'X-Api-Key': '041AA6FA66184165A38B8D938C68A30E',
}

# Add a dictionary to store the last position
last_position = {
    'x': 0,
    'y': 0,
    'z': 10,
}


def move(x=None, y=None, z=None, calibrate=False, home=False, speed=3000):
    global last_position
    if calibrate:
        json_data = {
            'command': f'G28'
        }
    elif home:
        json_data = {
            'command': f'G0 X{0} Y{240} Z{50} F1{3000}'
        }
        # Update the last sent position
        last_position['x'] = 0
        last_position['y'] = 240
        last_position['z'] = 50
    else:
        # Use the last sent position for unspecified axes
        x = last_position['x'] if x is None else x
        y = last_position['y'] if y is None else y
        z = last_position['z'] if z is None else z

        json_data = {
            'command': f'G0 X{x} Y{y} Z{z} F1{speed}'
        }

        # Update the last sent position
        last_position['x'] = x
        last_position['y'] = y
        last_position['z'] = z

    print(requests.post(
        'http://0.0.0.0/api/printer/command', headers=headers, json=json_data))


# move(calibrate=True)
# sleep(10)
move(home=True)
