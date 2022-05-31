from flask import Flask, render_template, request, flash, redirect, url_for,session
from flask_login import logout_user, current_user, login_user, login_required
from requests import Session
from app.models import User, NewsPrediction
from app import app,db
import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomTreesEmbedding
from xgboost import XGBRFClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import load
from app.prediction import make_predict,clean_text



@app.route('/',methods=['GET', 'POST'])
def index():
    
    if request.method == "POST":
        news = request.form.get('news')
        if news:
            newslist = news.split('\n')
            for nx in newslist:
                prediction = str(make_predict(nx)[0])
                session['last_prediction'] = prediction
                session['last_news'] = nx
                db.session.add(NewsPrediction(text=nx,prediction=prediction,user_id=session['id']))
            db.session.commit()
            flash('Prediction: {}'.format(prediction))
            return redirect(url_for('index'))
            pass
    if 'is_auth' in session and session['is_auth']:
        return render_template('index.html')
    return redirect('/login')

@app.route('/login',methods=['GET', 'POST'])
def login():
    if request.method=='POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username and password:
            user = User.query.filter_by(username=username).first()
            if user is None or not user.check_password(password = password):
                flash('Invalid username or password','danger')
                return redirect(url_for('login'))
            else:
                session['is_auth'] = True
                session['id'] = user.id
                session['user'] = user.username
                return redirect(url_for('index'))
        else:
            flash('Enter all details','danger')
            return redirect('/login')
    return render_template('login.html', title='Sign In')

@app.route('/forgot',methods=['GET', 'POST'])
def forgot():
    if request.method=='POST':
        email = request.form.get('email')
        if email:
            pass
    return render_template('forgot.html', title='Password reset page')

@app.route('/logout')
def logout():
    if 'is_auth' in session:
        session.pop('is_auth')
    return redirect(url_for('index'))


@app.route('/register',methods=['GET', 'POST'])
def register():
    if request.method=='POST':
        email = request.form.get('email')
        username = request.form.get('username')
        cpassword = request.form.get('cpassword')
        password = request.form.get('password')
        print(cpassword, password, cpassword==password)
        if username and password and cpassword and email:
            if cpassword != password:
                flash('Password do not match','danger')
                return redirect('/register')
            else:
                if User.query.filter_by(email=email).first() is not None:
                    flash('Please use a different email address','danger')
                    return redirect('/register')
                elif User.query.filter_by(username=username).first() is not None:
                    flash('Please use a different username','danger')
                    return redirect('/register')
                else:
                    user = User(username=username, email=email)
                    user.set_password(password)
                    db.session.add(user)
                    db.session.commit()
                    flash('Congratulations, you are now a registered user!','success')
                    return redirect(url_for('login'))
        else:
            flash('Fill all the fields','danger')
            return redirect('/register')
    return render_template('register.html', title='Sign Up page')


@app.route('/dashboard')
def dashboard():
    if 'is_auth' in session and not session['is_auth']:
        return redirect(url_for('login'))
    news = NewsPrediction.query.all()
    return render_template('dashboard.html', news=news)

@app.route('/news/<int:id>/delete')
def delete_news(id):
    if 'is_auth' in session and not session['is_auth']:
        return redirect(url_for('login'))
    news = NewsPrediction.query.get_or_404(id)
    db.session.delete(news)
    db.session.commit()
    if 'last_news' in session and session['last_news'] == news.text:
        session.pop('last_news')
    if 'last_prediction' in session and session['last_prediction'] == news.prediction:
        session.pop('last_prediction')
    flash('News deleted successfully','success')
    return redirect(url_for('dashboard'))


@app.route('/about')
def about():
    return render_template('about.html')

