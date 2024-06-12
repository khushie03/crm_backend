import os
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
query = input("Enter your query")
prompt="""Here you are acting as an Customer Relationship Management Chatbot. So on the basis of the database you need
to answer the queries of the user related to the data .
Provide detailed and best solutions for it and the data is """

def generate_gemini_content(transcript_text,prompt):

    model=genai.GenerativeModel("gemini-pro")
    response=model.generate_content(prompt + transcript_text)
    return response.text

print(generate_gemini_content(query,prompt + customer_details))