import os
from flask import Flask, render_template, Response, redirect, request
import cnn
import utils

app = Flask(__name__)

# ===== VIDEO FEED ROUTE =====


@app.route('/video_feed')
def video_feed():
    return Response(utils.detect_face(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/vidfeed_dataset/<user_id>')
def vidfeed_dataset(user_id):
    return Response(utils.generate_dataset(user_id), mimetype='multipart/x-mixed-replace; boundary=frame')


# ===== MAIN ROUTE =====

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/register_face/<user_id>')
def vfdataset_page(user_id):
    return render_template('register_face.html', user_id=user_id)


@app.route('/retrain')
def retrain():
    cnn.train_cnn()
    return redirect("/")


@app.route('/register')
def register():
    return render_template('register.html', user_id=utils.get_last_user_id() + 1)


@app.route('/register/<user_id>', methods=["POST"])
def register_post(user_id):
    name = request.form["name"]

    # Save to db
    utils.cur.execute(
        f'INSERT INTO users(user_id, name) VALUES({user_id}, "{name}")')
    utils.con.commit()

    return redirect("/")


if __name__ == '__main__':
    app.run(debug=os.environ["env"] != "prod", host='127.0.0.1', port=5000)
