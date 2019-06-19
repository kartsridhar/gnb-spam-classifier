from flask import Flask, render_template

# flask = Framework
# Flask = Python Class -> used to create instances of web applications

app = Flask(__name__)           # creates the instance of the class

@app.route('/')                 # function that is mapped to the URL
def home():
    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)             # debug = true to print out possible python errors
