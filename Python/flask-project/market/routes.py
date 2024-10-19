from market import app, db
from flask import render_template, redirect, url_for, flash
from market.models import Item, User
from market.forms import RegisterForm, LoginForm

@app.route('/')
@app.route('/home')
def home_page():
    return render_template("index.html")

@app.route('/market')
def market():
    items = Item.query.all()
    return render_template("market.html", items= items)

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        newuser = User(username=form.username.data,
                       email_address=form.email_address.data,
                       password=form.password1.data)
        db.session.add(newuser)
        db.session.commit()
        return redirect(url_for('market'))
    if form.errors != {}:
        for i in form.errors.values():
            flash(f"There was an error with creating a user: {i}", category='danger')
    return render_template('register.html', form = form)

@app.route("/login", methods = ['GET', 'POST'])
def login():
    form = LoginForm()
    return render_template("login.html", form = form)