from flask import Flask
from flask import jsonify, render_template

app = Flask(__name__)


@app.route('/hello/', methods=['GET'])
def hello():
    print ("hello")
    return render_template('test.html')

if __name__ == '__main__':
    app.run()