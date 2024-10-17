from market import app, db
from flask import render_template, redirect, url_for
from market.models import Item, User
from market.forms import RegisterForm

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
                       password_hash=form.password1.data)
        db.session.add(newuser)
        db.session.commit()
        return redirect(url_for('market'))
    if form.errors != {}:
        for i in form.errors.values():
            print(i)
    return render_template('register.html', form = form)