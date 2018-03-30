from flask import Flask
from flask import jsonify, render_template

app = Flask(__name__)


@app.route('/forcasts/', methods=['GET'])
def forcasts():
    print ("hello")
    return render_template('forcasts.html')


@app.route('/home/', methods=['GET'])
def home():
    print("home")
    return render_template('home.html')

if __name__ == '__main__':
    app.run()