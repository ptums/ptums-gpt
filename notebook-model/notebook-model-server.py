# Save this as server.py
from flask import Flask, request, jsonify
import torch
from transformers import BertTokenizer, BertModel

app = Flask(__name__)

# Load your model and tokenizer
'''
The purpose of these lines are to take your model
and make it understandable 
'''
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    texts = data['texts']
    
    # Tokenize and encode the texts
    encoded_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**encoded_inputs)
    
    # Extract the embeddings or whatever output you need
    embeddings = outputs.last_hidden_state.mean(dim=1).tolist()
    
    return jsonify(embeddings)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
