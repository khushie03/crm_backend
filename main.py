from flask import Flask, render_template, request, jsonify
from datetime import datetime
import mysql.connector
from send_message import send_message
import os
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications.resnet50 import preprocess_input
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import cv2
from recommender_code import image_similarity
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = Flask(__name__)

cnx = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Prusshita@1234",
    database="crm"
)

@app.route('/')
def index():
    return render_template("index.html")
@app.route('/login_status', methods=['GET'])
def login_details_analysis():
    try:
        cursor = cnx.cursor(dictionary=True)
        select_query = """
        SELECT status, COUNT(*) AS count
        FROM login_status
        GROUP BY status
        """
        cursor.execute(select_query)
        analysis_data = cursor.fetchall()
        cursor.close()
        
        return render_template("login_status.html", analysis_data=analysis_data)
    except mysql.connector.Error as err:
        print(f"Error in retrieving login details analysis: {err}")
        return jsonify({"error": "Error occurred while retrieving login details analysis."}), 500
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": "An error occurred while processing login details analysis."}), 500
    

@app.route('/client_data_Details', methods=['POST', 'GET'])
def client_Data_details():
    try:
        min_visits = request.args.get('number_of_visits', default=1, type=int)
        min_spend_rate = request.args.get('spend_rate', default=1.0, type=float)
        cursor = cnx.cursor(dictionary=True)
        select_query = """
        SELECT * FROM customer_details
        WHERE number_of_visits >= %s AND spend_rate >= %s
        """
        cursor.execute(select_query, (min_visits, min_spend_rate))
        customers = cursor.fetchall()
        cursor.close()
        logging.debug(customers)
        print(customers)
        print(min_visits)
        print(min_spend_rate)
        return render_template("customer_details.html", customers=customers)
    except mysql.connector.Error as err:
        logging.error(f"Error in retrieving items: {err}")
        return jsonify({"error": "Error occurred while retrieving customer details."})
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return jsonify({"error": "An error occurred while retrieving customer details."})
    


def insert_customer(name, number_of_visits, last_purchase_date, spend_rate, email_id):
    try:
        cursor = cnx.cursor()
        insert_query = """
        INSERT INTO customer_details (name, number_of_visits, last_purchase_date, spend_rate, email_id) 
        VALUES (%s, %s, %s, %s, %s)
        """
        cursor.execute(insert_query, (name, number_of_visits, last_purchase_date, spend_rate, email_id))
        cnx.commit()
        cursor.close()
        return 1
    except mysql.connector.Error as err:
        print(f"Error in inserting customer: {err}")
        cnx.rollback()
        return -1
    except Exception as e:
        print(f"An error occurred: {e}")
        cnx.rollback()
        return -1


@app.route('/insert_customer_route', methods=['POST'])
def insert_customer_route():
    if request.method == 'POST':
        name = request.form['name']
        number_of_visits = int(request.form['number_of_visits'])
        last_purchase_date = request.form['last_purchase_date']
        spend_rate = float(request.form['spend_rate'])
        email_id = request.form['email_id']
        
        result = insert_customer(name, number_of_visits, last_purchase_date, spend_rate, email_id)
        if result == 1:
            return render_template("successful_insertion.html")
        else:
            return "Error occurred while inserting customer data."
    else:
        return "Invalid request method."

    
@app.route('/signup')
def sign_up():
    return render_template("signup.html")

@app.route('/register', methods=['POST'])
def register():
    try:
        print("Form Data:", request.form)

        required_fields = ['name', 'age', 'dob', 'country', 'phone_number', 'password', 'usertype']
        for field in required_fields:
            if field not in request.form:
                return f"Missing field: {field}", 400
        name = request.form['name']
        age = request.form['age']
        dob = request.form['dob']
        country = request.form['country']
        phone_number = request.form['phone_number']
        password = request.form['password']
        usertype = request.form['usertype']

        result = insert_user(name, age, dob, country, phone_number, password, usertype)
        if result == 1:
            if usertype == 'Employee':
                return render_template("employee_dashboard.html", name=name)
            elif usertype == 'Client':
                return render_template("client_dashboard.html", name=name)
            elif usertype == 'Customer':
                return render_template("customer_dashboard.html", name=name)
            else:
                return render_template("generic_dashboard.html", name=name)
        else:
            return "Error occurred during registration."
    except Exception as e:
        print(f"An error occurred: {e}")
        return "An error occurred during registration.", 500

def insert_user(name, age, dob, country, phone_number, password, usertype):
    try:
        cursor = cnx.cursor()
        insert_query = """
        INSERT INTO login_details (name, age, dob, country, phone_number, password, usertype) 
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(insert_query, (name, age, dob, country, phone_number, password, usertype))
        cnx.commit()
        cursor.close()
        return 1
    except mysql.connector.Error as err:
        print(f"Error in inserting item: {err}")
        cnx.rollback()
        return -1
    except Exception as e:
        print(f"An error occurred: {e}")
        cnx.rollback()
        return -1

@app.route('/customer_dashboard', methods=['GET'])
def customer_dashboard():
    try:
        cursor = cnx.cursor(dictionary=True)
        select_query = """
        SELECT name, age, dob, country, phone_number, usertype
        FROM login_details
        """
        cursor.execute(select_query)
        login_data = cursor.fetchall()
        cursor.close()
        return render_template("login_details.html")
    except mysql.connector.Error as err:
        print(f"Error in retrieving items: {err}")
        return jsonify({"error": "Error occurred while retrieving login details."})
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": "An error occurred while retrieving login details."})

@app.route('/users', methods=['GET'])
def get_users():
    try:
        cursor = cnx.cursor(dictionary=True)
        select_query = "SELECT * FROM login_details"
        cursor.execute(select_query)
        users = cursor.fetchall()
        cursor.close()
        return jsonify(users)
    except mysql.connector.Error as err:
        print(f"Error in retrieving items: {err}")
        return jsonify({"error": "Error occurred while retrieving users."})
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": "An error occurred while retrieving users."})


@app.route('/send_email', methods=['POST'])
def send_email():
    data = request.json
    try:
        cursor = cnx.cursor(dictionary=True)
        select_query = """
        SELECT name, email_id FROM customer_details
        WHERE number_of_visits >= %s AND spend_rate >= %s
        """
        cursor.execute(select_query, (data.get('number_of_visits', 1), data.get('spend_rate', 1.0)))
        customers = cursor.fetchall()
        cursor.close()

        for customer in customers:
            send_message(customer['name'], customer['email_id'])
        
        return jsonify({"message": "Emails sent successfully!"}), 200
    except mysql.connector.Error as err:
        print(f"Error in retrieving items: {err}")
        return jsonify({"error": "Error occurred while retrieving customer details."}), 500
    except Exception as e:
        print(f"Error sending email: {e}")
        return jsonify({"error": "Failed to send email"}), 500
    

@app.route('/data_analytics', methods=['GET'])
def data_analytics():
    try:
        cursor = cnx.cursor(dictionary=True)
        select_query = """
        SELECT name, number_of_visits, spend_rate
        FROM customer_details
        """
        cursor.execute(select_query)
        customer_data = cursor.fetchall()
        cursor.close()

        customer_names = [customer['name'] for customer in customer_data]
        number_of_visits = [customer['number_of_visits'] for customer in customer_data]
        spend_rate = [customer['spend_rate'] for customer in customer_data]

        return render_template("data_analytics.html", 
                               customer_names=customer_names, 
                               number_of_visits=number_of_visits, 
                               spend_rate=spend_rate)
    except mysql.connector.Error as err:
        print(f"Error in retrieving customer data: {err}")
        return jsonify({"error": "Error occurred while retrieving customer data."})
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": "An error occurred while processing data analytics."})

def generate_gemini_content(transcript_text,prompt):

    model=genai.GenerativeModel("gemini-pro")
    response=model.generate_content(prompt + transcript_text)
    return response.text

def define_cloth_category(description):
    if pd.isna(description):
        return None

    keywords = ["heels", "dress", "scarf", "pants", "skirt", "shirt", "jacket","kaftaan","kafta", 
                "pant","suit","sari", "shorts", "purse", "kurti", "vest", "hoodie", "sneakers",
                "jumpsuit", "top", "sweater", "sunglasses", "sling", "shawl",
                "suit", "sash", "leli", "blouse", "tee", "lepa", "kurt", "cardigan",
                "sari", "sandals", "bag", "motifs", "lehen", "shaw", "rom", "pati",
                "stoler", "clutch", "rug", "dupa", "t-shirt", "saree", ""
                "anklet", "bangle", "bodysuit", "boots", "bra", "shoes", "coats", "t shirt",
                "sweater", "sweatshirt", "shirt", "jeans", "kurta", "dupatta", "coats", "coat",
                "scarf", "socks", "jacket", "blouse", "dress", "skirt", "pants", "leggings", "lera", "leaph",
                "gow", "rom", "kimono", "hoodie", ""
                "shorts", "suit", "tie", "gloves", "hat", "cap", "hoodie", "tunic", "tank top",
                "cardigan", "blazer", "vest", "poncho", "kimono", "sari", "jumpsuit", "romper",
                "overalls", "pajamas", "robe", "swimwear", "lingerie", "underwear", "nightwear",
                "sportswear", "activewear", "formalwear", "casualwear", "outerwear", "ethnicwear",
                "westernwear", "workwear", "uniform", "traditional", "vintage", "modern", "contemporary"]

    for keyword in keywords:
        if keyword in description.lower():
            return keyword
    return None
data = pd.read_csv("price.csv")
product_data = data[['id', 'price', 'description', 'category', 'code', 'href']]

tfidf_vectorizer = TfidfVectorizer(stop_words='english')
product_descriptions = product_data['description'].fillna('')
tfidf_matrix = tfidf_vectorizer.fit_transform(product_descriptions)

@app.route('/context_text', methods=['POST'])
def recommend_products_handler():
    user_choice = request.form['typed_text']
    text_category = define_cloth_category(user_choice)
    category_products = product_data[product_data['category'] == text_category]
    first_20_products = category_products.head(50)
    product_descriptions = first_20_products['description'].fillna('')
    user_vector = tfidf_vectorizer.transform([user_choice])
    product_vectors = tfidf_vectorizer.transform(product_descriptions)
    similarity_scores = np.dot(user_vector, product_vectors.T).toarray().flatten()
    top_indices = similarity_scores.argsort()[-50:][::-1]
    recommended_products = first_20_products.iloc[top_indices].to_dict(orient='records')
    return render_template('context_text.html', recommended_products=recommended_products)


def extract_features(image_path, model):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)  
    img = preprocess_input(img)  
    features = model.predict(img)
    return features.flatten() 

@app.route('/image_similarity', methods=['POST'])
def image_similarity():
    folder_path = r"C:\crm\static\Images"
    image_files = os.listdir(folder_path)[:30]  
    model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, pooling='avg')
    folder_features = {}
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        features = extract_features(image_path, model)
        folder_features[image_file] = features
    provided_image_file = request.files['image']
    provided_image_file.save('provided_image.jpg')
    provided_image_path = 'provided_image.jpg'
    provided_image_features = extract_features(provided_image_path, model)
    similarities = []
    for image_file, features in folder_features.items():
        similarity = cosine_similarity([provided_image_features], [features])[0][0]
        similarities.append({"id": image_file, "similarity_score": similarity})
    similar_images = sorted(similarities, key=lambda x: x['similarity_score'], reverse=True)[:10]
    return render_template('image_similarity.html', similar_images=similar_images)


@app.route('/image_context.html')
def image_finder():
    return render_template("image_context.html")

@app.route('/answer_question', methods=['POST', 'GET'])
def answer_question():
    try:
        cursor = cnx.cursor(dictionary=True)
        select_query = "SELECT * FROM customer_details"
        cursor.execute(select_query)
        customer_data = cursor.fetchall()
        cursor.close()
        customer_info = ""
        for customer in customer_data:
            customer_info += f"Customer Name: {customer['name']}, Number of Visits: {customer['number_of_visits']}, Spend Rate: {customer['spend_rate']}\n"
        prompt = """
        Here you are acting as a Customer Relationship Management Chatbot. So, on the basis of the database, you need to answer the queries of the user related to the data.
        Provide detailed and best solutions for it and the data is:\n
        """
        prompt += customer_info

        data = request.json
        question = data['question']

        
        answer = generate_gemini_content(question, prompt)

        return jsonify({"answer": answer}), 200
    except Exception as e:
        print(f"Error occurred while answering the question: {e}")
        return jsonify({"error": "An error occurred while fetching the answer."}), 500


if __name__ == '__main__':
    app.run(debug=True)
