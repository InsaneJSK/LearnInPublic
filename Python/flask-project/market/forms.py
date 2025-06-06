from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import Length, EqualTo, Email, DataRequired, ValidationError
from market.models import User

class RegisterForm(FlaskForm):

    def validate_username(self, nuser):
        user = User.query.filter_by(username=nuser.data).first()
        if user:
            raise ValidationError('Username already exists! Please try a different username')

    def validate_email(self, nemail):
        user = User.query.filter_by(email_address=nemail.data).first()
        if user:
            raise ValidationError('Email already registered! Please login instead')


    username = StringField(label= 'User Name: ', validators=[Length(min=2, max=30), DataRequired()])
    email_address = StringField(label= "Email Address: ", validators=[Email(), DataRequired()])
    password1 = PasswordField(label = "Password: ", validators=[Length(min=6), DataRequired()])
    password2 = PasswordField(label = "Confirm Password: ", validators=[EqualTo('password1'), DataRequired()])
    submit = SubmitField(label= 'Create Account')

class LoginForm(FlaskForm):
    username = StringField(label = 'User Name: ', validators = [DataRequired()])
    password = PasswordField(label = 'Password: ', validators = [DataRequired()])
    submit = SubmitField(label = 'Sign in')

class PurchaseItemForm(FlaskForm):
    submit = SubmitField(label = 'Purchase Item!')

class SellItemForm(FlaskForm):
    submit = SubmitField(label = 'Sell Item!')