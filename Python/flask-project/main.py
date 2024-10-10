from flask import Flask, render_template
app = Flask(__name__)

@app.route('/')
@app.route('/home')
def home_page():
    return render_template("index.html")

@app.route('/market')
def market():
    items = [
        {'id': 1, 'name': 'Phone', 'barcode': 123456789012, 'price': 500},
        {'id': 2, 'name': 'Laptop', 'barcode': 345678901234, 'price': 900},
        {'id': 3, 'name': 'Keyboard', 'barcode': 567890123456, 'price': 150}
    ]
    return render_template("market.html", items= items)

"""
Jinja Stuff to remember for future me:
when you need to access the value of a variable stored and passed in python, in html, use this: {{ var }}
for logic codes use {% logic %}
eg for loop 
{% for i in list_name %}
{% endfor %}
"""