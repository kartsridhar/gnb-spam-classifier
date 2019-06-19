from flask import Flask

# flask = Framework
# Flask = Python Class -> used to create instances of web applications

app = Flask(__name__)           # creates the instance of the class

@app.route('/hello')                 # function that is mapped to the URL
def home():
    return "Hello World!"

if __name__ == '__main__':
    app.run(debug=True)
