# now use llm
import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def generate_response(prompt):
    model = genai.GenerativeModel("gemini-3.1-flash-lite") # Select this specific Gemini model
    response = model.generate_content(prompt)  # generates response as per the parameters given
    return response.text
