"""This module creates the web-app to broadcast the camera output.

Module author: Mitch Miller <mitch.miller08@gmail.com>
2018-09-09

# TODO: Confirm response with state label works correctly

References:
    [1] Login authenticaiton with Flask.
        https://pythonspot.com/login-authentication-with-flask/

"""
import os
import pickle

from collections import Counter
from flask import Flask, render_template, Response
from flask import flash, redirect, request, session, abort

from camera import VideoCamera

app = Flask(__name__)
USERNAME = 'admin'
PASSWORD = 'password'

MODEL_PATH = 'model_training/trained_model_2018-08-31.h5'
LABEL_MAP_PATH = 'model_training/label_map_2018-08-31.pkl'
IDLE_INDEX = 0
NORMAL_INDEX = 1
## Number of states to consider (i.e. number of seconds)
STATE_LIMIT = 6
current_state = NORMAL_INDEX

with open(LABEL_MAP_PATH, 'rb') as label_file:
    label_map = pickle.load(label_file)


def get_common_state(q):
    state_list = []
    ## Get all items in queue
    while not q.empty():
        state_list.append(q.get())
    ## Return n most recent states to queue
    for state in state_list[-STATE_LIMIT:]:
        q.put(state)

    return Counter(states_list[-STATE_LIMIT:]).most_common()[0][0]

@app.route('/')
def index():
    if not session.get('logged_in'):
        return render_template('login.html')
    else:
        return render_template('index.html')

@app.route('/login', methods=['POST'])
def do_admin_login():
    if request.form['password'] == PASSWORD\
    and request.form['username'] == USERNAME:
        session['logged_in'] = True
    else:
        flash('Wrong password!')
    return index()

@app.route('/logout')
def logout():
    session['logged_in'] = False
    return index()

def gen(video_camera):
    while True:
        frame = video_camera.get_frame()
        current_state = get_common_state(video_camera.state_q)
        current_label = ' '.join(format(ord(x), 'b')
                                 for x in label_map[current_state])
        if current_state != IDLE_INDEX:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n'
                   b'Content-Type: text\r\n\r\n' + current_label +b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera(MODEL_PATH)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.secret_key = os.urandom(12)
    app.run(host='0.0.0.0', debug=True)
