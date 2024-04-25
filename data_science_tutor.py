from dotenv import load_dotenv
import streamlit as st
import os
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# Configure Google API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize GenerativeModel for AI Teaching Assistant
ai = genai.GenerativeModel(
    model_name="gemini-1.5-pro-latest",
    system_instruction=("""You are a helpful AI Teaching Assistant. Given an answer for the user query if you know, otherwise say "I don't know" if the user says Hi then respond with "Hi, this is Janvitha's chatbot. How can I help you?""")
)

# Initialize Gemini Pro model for chat
model = genai.GenerativeModel("gemini-pro") 
chat = model.start_chat(history=[])

# Initialize Streamlit app
st.set_page_config(page_title="Q&A Demo")
st.header("AI Data Science Tutor")

# Initialize session state for chat history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Text input and submit button
input_text = st.text_input("Input: ", key="input")
submit_button = st.button("Ask the question")

# Handle user input and display response
if submit_button and input_text:
    response = chat.send_message(input_text, stream=True)
    
    # Add user query and response to session state chat history
    st.session_state['chat_history'].append(("You", input_text))
    st.subheader("The Response is")
    for chunk in response:
        st.write(chunk.text)
        st.session_state['chat_history'].append(("Bot", chunk.text))

# Display chat history
st.subheader("The Chat History is")
for role, text in st.session_state['chat_history']:
    st.write(f"{role}: {text}")
