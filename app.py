# app.py
import sys
from flask import Flask, render_template, request, jsonify, Response
import json
import requests
import os
import logging
from datetime import datetime

app = Flask(__name__)

# Configure logging
logger = app.logger
formatter = logging.Formatter(fmt="%(asctime)s %(levelname)s %(module)s: %(message)s", datefmt="%H:%M:%S")
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
handler.setFormatter(formatter)
logger.handlers.clear()
logger.addHandler(handler)

# OPB API Configuration
OPB_API_BASE_URL = os.environ.get("OPB_API_URL", "http://0.0.0.0:8080")
API_KEY = os.environ["OPB_TEST_API_KEY"]
headers = {"X-API-KEY": API_KEY}


# Function to get user's location from IP
def get_user_location():
    """
    Get user's location based on their IP address
    Uses ipinfo.io API which has a free tier
    """
    try:
        # Get user's IP address
        if request.headers.getlist("X-Forwarded-For"):
            ip = request.headers.getlist("X-Forwarded-For")[0]
        else:
            ip = request.remote_addr
            
        # Skip for localhost testing
        if ip in ['127.0.0.1', 'localhost', '::1']:
            return None
            
        # Call IP info API
        response = requests.get(f'https://ipinfo.io/{ip}/json')
        if response.status_code == 200:
            data = response.json()
            # Return city and region/country if available
            location = data.get('city')
            region = data.get('region')
            country = data.get('country')
            
            if location and (region or country):
                if region:
                    location = f"{location}, {region}"
                elif country:
                    location = f"{location}, {country}"
                return location
            return None
    except Exception as e:
        logger.error(f"Error getting location from IP: {e}")
        return None

# Load legal tech providers
with open('legal_tech_providers.json', 'r') as f:
    providers = json.load(f)

# Configuration
BOT_ID = "48673b55-1e6c-45a3-afd3-dede7c0f5dc8"

# In-memory session storage
chat_sessions = {}

# Helper functions for API requests
def api_request(
    endpoint,
    method="POST",
    json=None,
    data=None,
    files=None,
    params=None,
    timeout=None,
    stream=None,
) -> requests.Response:
    """Make a request to the OPB API"""
    url = f"{OPB_API_BASE_URL}/{endpoint}"
    logger.info("Making %s request to /%s", method, endpoint)
    
    if method == "GET":
        return requests.get(url, headers=headers, data=data, json=json, params=params, timeout=timeout, stream=stream)
    if method == "DELETE":
        return requests.delete(url, headers=headers, data=data, json=json, params=params, timeout=timeout, stream=stream)
    return requests.post(url, headers=headers, data=data, json=json, files=files, params=params, timeout=timeout, stream=stream)

# Check API health
try:
    with api_request("", method="GET") as response:
        API_AVAILABLE = response.status_code == 200
        logger.info("Successfully connected to OPB API")
except Exception as e:
    logger.error(f"Failed to connect to OPB API: {e}")
    API_AVAILABLE = False

@app.route('/')
def index():
    return render_template('index.html', providers=providers)

@app.route('/api/initialize_chat', methods=['GET'])
def initialize_chat():
    """Initialize a new chat session with the backend"""
    
    try:
        # Initialize chat session via API
        with api_request("initialize_session", json={"bot_id": BOT_ID}) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("message") == "Success":
                    # Store the session ID
                    session_id = data.get("session_id")
                    chat_sessions[session_id] = {"started_at": datetime.now().isoformat()}

                    # Customize initial message based on location
                    initial_message = "Hello! I'm your legal tech directory assistant. How can I help you find the right legal technology solution? Please tell me about your legal issue, jurisdiction, and any other relevant factors."

                    # Get user's location
                    user_location = get_user_location()
                    if user_location:
                        chat_sessions[session_id]["location"] = user_location
                
                    return jsonify({
                        "session_id": session_id,
                        "initial_message": initial_message,
                        "detected_location": user_location,
                    })
            
            # If we get here, something went wrong
            logger.error(f"Error initializing chat: {response.text if hasattr(response, 'text') else 'Unknown error'}")
            return jsonify({"error": "Failed to initialize chat session with backend API"}), 500
    except Exception as e:
        logger.error(f"Error initializing chat: {e}")
        return jsonify({"error": "Failed to connect to backend API"}), 503

@app.route('/api/update_location', methods=['POST'])
def update_location():
    session_id = request.json.get('session_id')
    new_location = request.json.get('location')
    
    if session_id in chat_sessions:
        chat_sessions[session_id]['location'] = new_location
        logger.info("Location updated")
        return jsonify({"status": "Location updated"})
    return jsonify({"error": "Invalid session"}), 400

@app.route('/api/chat_stream', methods=['GET'])
def chat_stream():
    """Stream chat responses for a more interactive experience"""
    user_message = request.args.get('message', '')
    session_id = request.args.get('session_id')
    
    if not session_id or session_id not in chat_sessions:
        return jsonify({"error": "Invalid session ID"}), 400
    
    session = chat_sessions[session_id] if session_id in chat_sessions else None
    
    def generate():
        # Check if we have an OPB session ID (API integration)
        if session is not None:
            try:
                # Create a streaming request to the OPB API
                # Using requests.get with stream=True for SSE
                with api_request(
                    "chat_session_stream",
                    json={
                        "session_id": session_id,
                        "message": user_message + (f" I am in {session['location']}." if session.get('location') else ""),
                        "user": {"email": "test@test.com", "firebase_uid": "idk"}
                    },
                    stream=True
                ) as r:
                    if r.status_code == 200:
                        # Process the streamed response
                        complete_response = ""
                        in_solution_evaluation = False
                        solution_evaluation_buffer = ""
                        
                        for line in r.iter_lines(decode_unicode=True):
                            if line:
                                try:
                                    data_json = json.loads(line)
                                    
                                    if data_json.get("type") == "response":
                                        content = data_json.get("content", "")
                                        
                                        # Handle solution_evaluation streaming
                                        if "<solution_evaluation>" in content or in_solution_evaluation:
                                            if "<solution_evaluation>" in content and not in_solution_evaluation:
                                                # Start of solution_evaluation
                                                in_solution_evaluation = True
                                                start_idx = content.find("<solution_evaluation>")
                                                before_tag = content[:start_idx]
                                                after_tag = content[start_idx:]
                                                
                                                if before_tag.strip():
                                                    data_json["content"] = before_tag
                                                    yield f"data: {json.dumps(data_json)}\n\n"
                                                
                                                solution_part = after_tag[:after_tag.find("<solution_evaluation>") + len("<solution_evaluation>")]
                                                solution_evaluation_buffer += solution_part
                                                content = after_tag[len(solution_part):]

                                            # Finalize on <response> detection
                                            if "<response>" in content and in_solution_evaluation:
                                                split_idx = content.find("<response>")
                                                solution_part = content[:split_idx]
                                                solution_evaluation_buffer += solution_part
                                                
                                                # Send final solution evaluation chunk
                                                yield f"data: {{\"type\": \"solution_evaluation\", \"content\": {json.dumps(solution_evaluation_buffer)}}}\n\n"
                                                solution_evaluation_buffer = ""
                                                in_solution_evaluation = False
                                                
                                                # Process the remaining content
                                                data_json["content"] = content[split_idx:]
                                                yield f"data: {json.dumps(data_json)}\n\n"
                                                continue

                                            # Regular solution evaluation processing
                                            if in_solution_evaluation:
                                                if "</solution_evaluation>" in content:
                                                    split_idx = content.find("</solution_evaluation>")
                                                    solution_part = content[:split_idx]
                                                    if solution_part.strip():
                                                        yield f"data: {{\"type\": \"solution_evaluation\", \"content\": {json.dumps(solution_part)}}}\n\n"

                                                    # Process remaining content after the tag
                                                    split_idx += len("</solution_evaluation>")
                                                    remaining = content[split_idx:]
                                                    if remaining.strip():
                                                        data_json["content"] = remaining
                                                        yield f"data: {json.dumps(data_json)}\n\n"

                                                    in_solution_evaluation = False
                                                    solution_evaluation_buffer = ""  # Reset the buffer
                                                else:
                                                    # Yield only the current content
                                                    yield f"data: {{\"type\": \"solution_evaluation\", \"content\": {json.dumps(content)}}}\n\n"
                                                    solution_evaluation_buffer += content
                                        
                                        # Handle structured response tags
                                        elif any(tag in content for tag in ["<response>", "</response>", "<summary>", "</summary>"]):
                                            # Existing structured response handling
                                            data_json["is_structured"] = True
                                            yield f"data: {json.dumps(data_json)}\n\n"
                                        
                                        else:
                                            # Regular response content
                                            yield f"data: {line}\n\n"
                                    elif data_json.get("type") == "done":
                                        # Finalize solution_evaluation if unclosed
                                        if in_solution_evaluation and solution_evaluation_buffer:
                                            yield f"data: {{\"type\": \"solution_evaluation\", \"content\": {json.dumps(solution_evaluation_buffer)}}}\n\n"

                                        # Parse structured response parts if present
                                        response_parts = {}
                                        # Check if we have a structured response using <response> tag
                                        if "<response>" in complete_response and "</response>" in complete_response:
                                            response_content = complete_response[complete_response.find("<response>") + len("<response>"):
                                                                             complete_response.find("</response>")]
                                            
                                            # Extract each section within the response
                                            for section in ["summary", "recommendations", "disclaimer", "follow_up_questions"]:
                                                if f"<{section}>" in response_content and f"</{section}>" in response_content:
                                                    start_tag = f"<{section}>"
                                                    end_tag = f"</{section}>"
                                                    content = response_content[response_content.find(start_tag) + len(start_tag):
                                                                              response_content.find(end_tag)]
                                                    response_parts[section] = content.strip()
                                        # For backward compatibility, also check for sections directly
                                        else:
                                            for section in ["summary", "recommendations", "disclaimer", "follow_up_questions"]:
                                                if f"<{section}>" in complete_response and f"</{section}>" in complete_response:
                                                    start_tag = f"<{section}>"
                                                    end_tag = f"</{section}>"
                                                    content = complete_response[complete_response.find(start_tag) + len(start_tag):
                                                                               complete_response.find(end_tag)]
                                                    response_parts[section] = content.strip()
                                                    
                                        # Check if we found any structured parts
                                        if len(response_parts) == 0:
                                            # No structured parts found, include the full response for backward compatibility
                                            # Remove any XML tags remaining in the response
                                            clean_response = complete_response
                                            for tag in ["response", "summary", "recommendations", "disclaimer", "follow_up_questions"]:
                                                clean_response = clean_response.replace(f"<{tag}>", "").replace(f"</{tag}>", "")
                                            response_parts["summary"] = clean_response.strip()
                                        
                                        # Send very simple done event with just the type and content
                                        # We'll include the structured response directly rather than trying to parse it
                                        yield f"data: {{\"type\": \"done\", \"content\": {json.dumps(complete_response)}}}\n\n"
                                    else:
                                        # Forward other message types
                                        yield f"data: {line}\n\n"
                                except json.JSONDecodeError:
                                    logger.error(f"Invalid JSON in SSE data: {line}")
                                    continue
                    else:
                        logger.error(f"Error in streaming API: {r.status_code} | {r.text}")
                        yield f"data: {{\"type\": \"error\", \"message\": \"Failed to connect to backend API (status {r.status_code})\"}}\n\n"
            except Exception:
                logger.exception("Error in chat stream.")
                yield "data: {\"type\": \"error\", \"message\": \"Failed to connect to backend API\"}\n\n"
        else:
            # No OPB session ID
            yield "data: {\"type\": \"error\", \"message\": \"Invalid chat session\"}\n\n"
    
    return Response(generate(), content_type='text/event-stream')

@app.route('/api/search', methods=['POST'])
def search():
    """Search for legal tech providers based on product name and filters"""
    query = request.json.get('query', '').strip()
    filters = request.json.get('filters', {})
    
    # Start with all providers
    results = providers.copy()
    
    # If query is provided, filter by product name
    if query:
        # Simple product name matching
        results = [p for p in results if query.lower() in p['product_name'].lower()]
        
        # If no results found with exact product name match, try more flexible matching
        if not results:
            results = [p for p in providers if calculate_relevance(query, p) > 0.3]
    
    # Apply additional filters
    results = apply_filters(results, filters)

    return jsonify({'results': results})

def apply_filters(results, filters):
    """Apply filters to search results"""
    filtered_results = results.copy()
    
    # Filter by problem that product solves (currently labeled as 'problem' in the frontend)
    if filters.get('problem') and filters['problem'] != "Any":
        filtered_results = [p for p in filtered_results if 
                           filters['problem'] in p['problem_that_product_solves'].split(',')]
    
    # Filter by service area
    if filters.get('service_area') and filters['service_area'] != "":
        filtered_results = [p for p in filtered_results if 
                           filters['service_area'] in p['service_area'].split(',')]
    
    # Filter by pricing model
    if filters.get('pricing') and filters['pricing'] != "Any Price":
        filtered_results = [p for p in filtered_results if 
                          p['pricing'] == filters['pricing']]
    
    # Filter by mobile app availability
    if filters.get('mobile_app_available') and filters['mobile_app_available'] != "Any":
        filtered_results = [p for p in filtered_results if 
                          p['mobile_app_available'] == filters['mobile_app_available']]
    
    # Filter by Justice Tech Association membership
    if filters.get('justice_tech_assn_member') and filters['justice_tech_assn_member'] != "Any":
        filtered_results = [p for p in filtered_results if 
                          p['justice_tech_assn_member'] == filters['justice_tech_assn_member']]
    
    return filtered_results

@app.route('/api/provider/<int:provider_id>', methods=['GET'])
def get_provider(provider_id):
    """Get detailed information about a specific provider"""
    provider = next((p for p in providers if p['id'] == provider_id), None)
    
    if not provider:
        return jsonify({"error": "Provider not found"}), 404
    
    # In a real implementation, you might add additional details here
    return jsonify(provider)

def calculate_relevance(query, provider, embeddings=None):
    """Calculate relevance score between query and provider with focus on product names"""
    query_lower = query.lower()
    
    # For product name search, prioritize exact or partial product name matches
    product_name_lower = provider['product_name'].lower()
    
    # Exact product name match (highest score)
    if query_lower == product_name_lower:
        return 1.0
    
    # Product name starts with query
    if product_name_lower.startswith(query_lower):
        return 0.9
    
    # Query is a significant part of the product name
    if query_lower in product_name_lower:
        return 0.8
    
    # Partial match - check if words in query match parts of product name
    query_words = query_lower.split()
    product_words = product_name_lower.split()
    
    # Count matching words
    matching_words = sum(1 for word in query_words if any(word in pw for pw in product_words))
    if matching_words > 0:
        return 0.6 * (matching_words / len(query_words))
    
    # Very low relevance for non-product name matches
    return 0.1

if __name__ == '__main__':
    app.run(debug=True)