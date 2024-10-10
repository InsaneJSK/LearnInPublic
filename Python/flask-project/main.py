from flask import Flask, render_template
app = Flask(__name__)

@app.route('/')
@app.route('/home')
def home_page():
    return render_template("index.html")

@app.route('/market')
def market():
    return render_template("market.html")

"""
@app.route('/about')
def about_page():
    return "<h1>About page!</h1>"

@app.route('/about/<username>')
def about_user(username):
    return f"<h1>This is the about page of {username}"
"""