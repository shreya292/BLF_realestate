#importing importat module for set up flask application

from flask import Flask, render_template, request, url_for,redirect,flash,g, session
from flask_session import Session
from flask_bcrypt import Bcrypt              #for generate, hash, check the password
import secrets                               #generate a secure secret key 
import sqlite3
import pickle
import pandas as pd


#------------------------------------------------------------------------------------------------------------------#

#creating Flask web app
app= Flask(__name__)       #create an object of Flask class(constructor) for web application

#load pickel file ( Load the pre-trained machine learning model)
model = pickle.load(open("module.pkl", "rb"))


#In Flask, the flash function is used to display a message to the user, and it requires the SECRET_KEY to be set in the Flask application configuration. The SECRET_KEY is used for encrypting the messages before storing them in the client's cookies.When the flash function is called, the message is stored in a special Flask session cookie, which is signed with the SECRET_KEY. When the user makes a subsequent request, the message is retrieved from the session cookie, decrypted using the SECRET_KEY, and displayed to the user.Therefore, the SECRET_KEY is an important security feature in Flask, and it should be kept secret and not shared with others. It's used to protect sensitive data, like the messages passed to the flash function, from being tampered with or viewed by unauthorized users.It is important to note that changing the secret key will invalidate all existing sessions.

#configaration
app.secret_key = secrets.token_hex(16)                  #This generates a 16-byte random hex string as the secret key.

#initializing important Flask extensions in the app:
bcrypt = Bcrypt(app)        # initializes the Flask-Bcrypt extension which provides secure password hashing utilities.

# database configuration
app.config['DATABASE'] = 'user.db'
app.config["SESSION_PERMANENT"]=False
app.config["SESSION_TYPE"]='filesystem'

Session(app)

#returns a SQLite database connection (global)
def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(app.config['DATABASE'])
    return db

#This code is a Flask app context processor that automatically closes the database connection after each request is processed. The @app.teardown_appcontext decorator registers a function to be called when the application context is torn down. In this case, the function checks if a database connection is open, and if so, it closes it.
@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

#----------------------------------------------Main page------------------------------------------------------

#Routing for one single web page
#routing for main page

@app.route('/')     #create a route for root url for showing the main page 
def main_page():
    return render_template('index.html')

#------------------------------------------HOME PAGE---------------------------------------------------------------#
#routing for home section

@app.route('/home')
def home():
    return render_template('index.html', section='#home')

#Make An Enquiry(price page)
@app.route('/house' , methods=['GET', 'POST'])
def house():
    if request.method == 'GET':
        return render_template('home.html')

    if request.method == 'POST':
        
    #For rendering results on HTML GUI
    # Retrieve the house features entered by the user from the form data
        feature1 = request.form['OverallQual']
        feature2 = request.form['ExterQual']
        feature3 = request.form['BsmtQual']
        feature4 = request.form['LotArea']
        feature5 = request.form['GrLivArea']
        feature6 = request.form['GarageArea']
        feature7 = request.form['TotalBsmtSF']
        feature8 = request.form['TotRmsAbvGrd']
        feature9 =request.form['YearBuilt']
        #feature10 = request.form['SalePrice']
    
    
        # Create a DataFrame with the input variables
        input_variables = pd.DataFrame([[feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9]], columns=['OverallQual', 'ExterQual', 'BsmtQual', 'LotArea', 'GrLivArea',
                                                    'GarageArea', 'TotalBsmtSF', 'TotRmsAbvGrd','YearBuilt'])

        # Make the prediction
        prediction = model.predict(input_variables)
        predicted_price = prediction[0][0]  # Extract the predicted value
        
        output= round(predicted_price,2)
        predict = f"Estimated Price: Rs {output}"

         # Render the template with the result
        return render_template('home.html',
                               original_input={'OverallQual': feature1, 'ExterQual': feature2, 'BsmtQual': feature3,
                                               'LotArea': feature4, 'GrLivArea': feature5, 'GarageArea': feature6,
                                               'TotalBsmtSF': feature7, 'TotRmsAbvGrd': feature8,'YearBuilt':feature9 },
                               result=predict)

#-----------------------------------------------------------------------------------------------------------------#

#routing for about section

@app.route('/about')
def about():
    return render_template('index.html', section='#about')

#routing for service section

@app.route('/service')
def service():
    return render_template('index.html', section='#service')

#routing for facilities section

@app.route('/facilities')
def facilities():
    return render_template('index.html' , section='#facilities')

#routing for designer section

@app.route('/designer')
def designer():
    return render_template('index.html' , section='#designer')

#routing for contact section

@app.route('/contact')
def contact():
    return render_template('index.html', section='#contact')

#store the data from the contact us form which is given by user to the database.
#https://extendsclass.com/sqlite-browser.html#

@app.route('/contact', methods=['GET', 'POST'])
def submit_form():
    
    #check if the request method is POST.
    
    if request.method == 'POST':
        
        #extract the form data using the request.form dictionary and insert it into the database using SQLite 
        
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']

        # Store data in the database
        conn = sqlite3.connect('user.db')
        c = conn.cursor()
        c.execute('CREATE TABLE IF NOT EXISTS User_Info (name TEXT , email TEXT, message TEXT)')
        c.execute("INSERT INTO User_Info (name, email, message) VALUES (?, ?, ?)", (name, email, message))
        conn.commit()
        conn.close()

        #If the data is successfully stored in the database, redirect the user to a new page that displays a success message. 
        
        flash('Your message was successfully submitted!')  #flash() function in Flask to store the message and display it on the new page.
        
        return redirect(url_for('submit_form'))
    
    return render_template('index.html')

# When the form is submitted, the data is extracted from the request.form dictionary and stored in an SQLite database using the sqlite3 module. If the data is successfully stored, the user is redirected back to the form page with a success message using the Flask flash() function. The success message is displayed on the HTML page using the Flask get_flashed_messages() function.

if __name__ == '__main__':
    app.run(debug=True)