# app.py
from flask import Flask, render_template, request, jsonify
import json

app = Flask(__name__)

# Mock database of legal tech providers
with open('legal_tech_providers.json', 'r') as f:
    providers = json.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')
    
    # Simple mock response - in production, this would call your LLM API
    response = {
        'message': "I'll help you find legal tech solutions based on your needs. Could you tell me more about your legal issue and your budget?",
        'suggested_filters': ['jurisdiction:US', 'category:Contract Management']
    }
    
    return jsonify(response)

@app.route('/api/search', methods=['POST'])
def search():
    query = request.json.get('query', '')
    filters = request.json.get('filters', {})
    
    # Mock search function - in production, this would query your database
    results = [p for p in providers if query.lower() in p['name'].lower()]
    
    # Apply filters
    if 'jurisdiction' in filters:
        results = [p for p in results if filters['jurisdiction'] in p['jurisdictions']]
    if 'category' in filters:
        results = [p for p in results if filters['category'] in p['categories']]
    if 'max_price' in filters:
        results = [p for p in results if p['price_tier'] <= filters['max_price']]
    
    return jsonify({'results': results})

if __name__ == '__main__':
    app.run(debug=True)