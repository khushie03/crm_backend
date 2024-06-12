# CRM Backend: Customer Engagement Redefined

Welcome to CRM Backend, where customer relationship management meets cutting-edge technology to enhance customer engagement and operational efficiency. This Flask-powered application seamlessly integrates with MySQL for robust data management and leverages advanced machine learning models for intelligent insights and recommendations.

## Key Features

- **Login Status Analysis**: Gain actionable insights into user login behaviors and patterns.
  
- **Customer Data Details**: Effortlessly filter and display customer data based on customizable criteria.

- **User Registration**: Register new users with dynamic roles (Employee, Client, Customer) and streamline access management.

- **Email Automation**: Automate personalized email campaigns to nurture customer relationships effectively.

- **Data Analytics**: Visualize and analyze customer behavior metrics to drive informed business decisions.

- **Product Recommendation**: Harness the power of AI to recommend tailored products based on customer preferences.

- **Image Similarity**: Enable visual search capabilities by finding similar images using advanced image processing techniques.

- **Customer Query Handling**: Empower your CRM with AI-driven chatbot capabilities to respond intelligently to customer queries.

## Setup and Installation

### Prerequisites

- Python 3.x
- Flask
- MySQL
- TensorFlow
- OpenAI API key for generative AI
- Required Python packages (see `requirements.txt`)

### Getting Started

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/khushie03/crm_backend.git
   cd crm_backend
   ```

2. **Setup Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**:
   Create a `.env` file in the root directory:
   ```dotenv
   GOOGLE_API_KEY=your_google_api_key
   ```

5. **Database Configuration**:
   - Create a MySQL database named `crm`.
   - Update database connection details in `app.py`:
     ```python
     cnx = mysql.connector.connect(
         host="localhost",
         user="root",
         password="your_password",
         database="crm"
     )
     ```

6. **Run the Application**:
   ```bash
   python app.py
   ```

7. **Access the Application**:
   Open your web browser and navigate to `http://localhost:5000`.

## Directory Structure

- `templates/`: HTML templates for frontend presentation.
- `static/`: Static assets (images, CSS, JS).
- `app.py`: Core Flask application file.
- `requirements.txt`: Dependencies list.

## API Endpoints

- **Home**: `GET /`
- **Login Status Analysis**: `GET /login_status`
- **Customer Data Details**: `GET /client_data_Details`, `POST /client_data_Details`
- **User Registration**: `GET /signup`, `POST /register`
- **Data Analytics**: `GET /data_analytics`
- **Send Emails**: `POST /send_email`
- **Product Recommendation**: `POST /context_text`
- **Image Similarity**: `POST /image_similarity`
- **Answer Customer Queries**: `POST /answer_question`

## Additional Features

- **Helper Functions**:
  - `insert_customer(name, number_of_visits, last_purchase_date, spend_rate, email_id)`
  - `insert_user(name, age, dob, country, phone_number, password, usertype)`
  - `extract_features(image_path, model)`
  - `define_cloth_category(description)`
  - `generate_gemini_content(transcript_text, prompt)`

## Contribute

Contributions are welcome! Fork the repository, make your changes, and submit a pull request. Let's innovate together!

## License

This project is licensed under the MIT License.

---

Feel free to further customize the README to include specific details about your project's architecture, unique selling points, or any other information that highlights the creativity and innovation of your CRM backend application.
