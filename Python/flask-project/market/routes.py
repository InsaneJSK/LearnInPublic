from market import app
from flask import render_template
from market.models import Item
from market.forms import RegisterForm

@app.route('/')
@app.route('/home')
def home_page():
    return render_template("index.html")

@app.route('/market')
def market():
    items = Item.query.all()
    return render_template("market.html", items= items)

@app.route('/register')
def register():
    form = RegisterForm()
    return render_template('register.html', form = form)