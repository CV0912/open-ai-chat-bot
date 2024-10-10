from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import openai
import json
import os

app = Flask(__name__)
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
openai.api_key = 'api key'
CONVERSATION_FILE = 'conversation_history.json'

def load_convo():
    if os.path.exists(CONVERSATION_FILE) and os.path.getsize(CONVERSATION_FILE) > 0:
        with open(CONVERSATION_FILE, 'r') as file:
            try:
                return json.load(file) or []
            except json.JSONDecodeError:
                pass 
    return []  # Return an empty list if the file doesn't exist or is empty

def save_conversation(conversation_history):
    with open("conversation_history.json", 'w') as file:
        json.dump(conversation_history, file)
def query_openai(prompt, conversation_history):
    message = [{"role": "system", "content": ""}]
    # Include past conversation to keep context
    message.extend(conversation_history)
    message.append({"role": "user", "content": prompt})
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=message,  # Combine question with context
        max_tokens=150,
        n=1  # Request 1 completion
    )
    answers = response.choices[0].message.content.strip()
    return answers
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    prompt = data.get('prompt', '')  
    context = data.get('context', '') 
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400
    conversation_history = load_convo()
    answer = query_openai(prompt, conversation_history)
    conversation_history.append({"role": "user", "content": prompt})
    conversation_history.append({"role": "assistant", "content": answer})
    save_conversation(conversation_history)

    return jsonify({'response': answer})

# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True)  