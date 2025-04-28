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
OPB_API_BASE_URL = os.environ.get("OPB_API_BASE_URL", "http://0.0.0.0:8080")
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
BOT_ID = "1d43728f-3673-4817-a0d9-f1271435e11c"

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
                    
                    # Get user's location
                    user_location = get_user_location()
                    
                    # Customize initial message based on location
                    initial_message = "Hello! I'm your legal tech directory assistant. How can I help you find the right legal technology solution? Please tell me about your legal issue, jurisdiction, and budget."
                    
                    # Add location info if available
                    if user_location:
                        initial_message = f"It looks like you're in {user_location}. If this is incorrect, let me know! " + initial_message
                    
                    return jsonify({
                        "session_id": session_id,
                        "initial_message": initial_message
                    })
            
            # If we get here, something went wrong
            logger.error(f"Error initializing chat: {response.text if hasattr(response, 'text') else 'Unknown error'}")
            return jsonify({"error": "Failed to initialize chat session with backend API"}), 500
    except Exception as e:
        logger.error(f"Error initializing chat: {e}")
        return jsonify({"error": "Failed to connect to backend API"}), 503

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
                        "message": user_message,
                        "user": {"email": "test@test.com", "firebase_uid": "idk"}
                    },
                    stream=True
                ) as r:
                    if r.status_code == 200:
                        # Process the streamed response
                        complete_response = ""
                        in_query_analysis = False
                        current_section = None
                        section_content = {}
                        query_analysis_buffer = ""
                        
                        for line in r.iter_lines(decode_unicode=True):
                            if line:
                                logger.info("Received line from API: %s", line)
                                try:
                                    data_json = json.loads(line)
                                    
                                    if data_json.get("type") == "response":
                                        content = data_json.get("content", "")
                                        # Accumulate the complete response
                                        complete_response += content
                                        
                                        # Check for and handle query analysis tags - these should be filtered out
                                        if "<query_analysis>" in content or in_query_analysis:
                                            if "<query_analysis>" in content and not in_query_analysis:
                                                in_query_analysis = True
                                                query_analysis_buffer = content[content.find("<query_analysis>"):]
                                                clean_content = content[:content.find("<query_analysis>")]
                                            elif "</query_analysis>" in content and in_query_analysis:
                                                in_query_analysis = False
                                                query_analysis_buffer += content[:content.find("</query_analysis>") + len("</query_analysis>")]
                                                clean_content = content[content.find("</query_analysis>") + len("</query_analysis>"):]
                                            elif in_query_analysis:
                                                query_analysis_buffer += content
                                                clean_content = ""
                                            else:
                                                clean_content = content
                                                
                                            # Only send non-empty, clean content to the client
                                            if clean_content.strip():
                                                data_json["content"] = clean_content
                                                yield f"data: {json.dumps(data_json)}\n\n"
                                        # Check if this is part of the structured response - we'll just accumulate, not display directly
                                        # as it will be processed and displayed properly when "done" is received
                                        elif "<response>" in content or "</response>" in content or "<summary>" in content or "</summary>" in content or "<recommendations>" in content or "</recommendations>" in content or "<disclaimer>" in content or "</disclaimer>" in content or "<follow_up_questions>" in content or "</follow_up_questions>" in content:
                                            # For structured response parts, we'll just send a special indicator
                                            # The actual content will still be accumulated in complete_response
                                            # But we'll set is_structured flag to inform the frontend this is structured content
                                            data_json["is_structured"] = True
                                            
                                            # Always keep the original content so the client can accumulate it
                                            # for proper extraction of sections later if needed
                                            # The client will just show the loading indicator while accumulating
                                                
                                            yield f"data: {json.dumps(data_json)}\n\n"
                                        else:
                                            # Forward the chunk to the client as is
                                            yield f"data: {line}\n\n"
                                    elif data_json.get("type") == "done":
                                        # When done, add suggested filters and structure the content
                                        # First remove any query analysis from the complete response
                                        if "<query_analysis>" in complete_response and "</query_analysis>" in complete_response:
                                            query_analysis = complete_response[complete_response.find("<query_analysis>"):complete_response.find("</query_analysis>") + len("</query_analysis>")]
                                            complete_response = complete_response.replace(query_analysis, "")
                                            
                                        suggested_filters = extract_filters_from_response(complete_response, user_message)
                                        
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
                                        logger.info("Sending done event - simplified approach")
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
    """Search for legal tech providers based on query and filters"""
    query = request.json.get('query', '')
    filters = request.json.get('filters', {})
    
    if query and API_AVAILABLE:
        try:
            # Use OPB's API for embeddings
            with api_request(
                "encoders/encode_text",
                json={"text": query}
            ) as response:
                if response.status_code == 200:
                    data = response.json()
                    if data.get("success"):
                        embeddings = data.get("embeddings", [])
                        
                        # We now have embeddings, use them to calculate semantic similarity
                        # This would normally be done by the backend, but for this demo
                        # we'll use a simplified approach
                        results = []
                        
                        for provider in providers:
                            # Calculate relevance using embeddings and text matching
                            relevance = calculate_relevance(query, provider, embeddings)
                            if relevance > 0.3:  # Lower threshold for more results
                                provider_copy = provider.copy()
                                provider_copy['relevance'] = relevance
                                results.append(provider_copy)
                        
                        # Sort by relevance
                        results.sort(key=lambda x: x['relevance'], reverse=True)
                    else:
                        # API error
                        logger.error(f"Error in embeddings API: {data.get('error', 'Unknown error')}")
                        return jsonify({"error": "Failed to process search query"}), 500
                else:
                    # API call failed
                    logger.error(f"Error in embeddings API call: {response.status_code} - {response.text if hasattr(response, 'text') else 'Unknown error'}")
                    return jsonify({"error": "Failed to process search query"}), 500
                    
        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            return jsonify({"error": "Failed to connect to backend API"}), 503
    else:
        # No query or API unavailable
        if not API_AVAILABLE:
            return jsonify({"error": "Search service unavailable"}), 503
        else:
            # Empty query, return all providers
            results = providers.copy()
    
    # Apply filters
    results = apply_filters(results, filters)
    
    return jsonify({'results': results})


def apply_filters(results, filters):
    """Apply filters to search results"""
    if filters.get('jurisdiction'):
        results = [p for p in results if filters['jurisdiction'] in p['jurisdiction']]
    
    if filters.get('service_area'):
        results = [p for p in results if filters['service_area'] == p['service_area']]
    
    if filters.get('pricing'):
        results = [p for p in results if filters['pricing'] == p['pricing']]
    
    if filters.get('mobile_app_available'):
        results = [p for p in results if filters['mobile_app_available'] == p['mobile_app_available']]
    
    if filters.get('justice_tech_assn_member'):
        results = [p for p in results if filters['justice_tech_assn_member'] == p['justice_tech_assn_member']]
    
    return results

@app.route('/api/provider/<int:provider_id>', methods=['GET'])
def get_provider(provider_id):
    """Get detailed information about a specific provider"""
    provider = next((p for p in providers if p['id'] == provider_id), None)
    
    if not provider:
        return jsonify({"error": "Provider not found"}), 404
    
    # In a real implementation, you might add additional details here
    return jsonify(provider)

def extract_filters_from_response(response, user_message):
    """Extract suggested filters from the assistant's response"""
    suggested_filters = []
    
    # Check for jurisdictions
    jurisdictions = ['US', 'UK', 'EU', 'CA', 'AU']
    for jurisdiction in jurisdictions:
        if jurisdiction.lower() in response.lower() or jurisdiction.lower() in user_message.lower():
            suggested_filters.append(f"jurisdiction:{jurisdiction}")
            break
    
    # Check for service areas
    service_areas = [
        'Document Automation', 
        'Contract Management', 
        'Practice Management', 
        'E-Discovery', 
        'Legal Research',
        'Client Intake',
        'Compliance'
    ]
    for service_area in service_areas:
        if service_area.lower() in response.lower() or service_area.lower() in user_message.lower():
            suggested_filters.append(f"service_area:{service_area}")
            break
    
    # Check for pricing indicators
    pricing_terms = {
        'free': 'Free',
        'subscription': 'Subscription',
        'contact': 'Contact for quote',
        'flat fee': 'Flat + Fees',
        'yearly': 'Yearly'
    }
    
    for term, pricing in pricing_terms.items():
        if term.lower() in response.lower() or term.lower() in user_message.lower():
            suggested_filters.append(f"pricing:{pricing}")
            break
    
    # If no specific pricing found but price is mentioned, suggest checking all options
    if ('price' in response.lower() or 'cost' in response.lower() or 
        'price' in user_message.lower() or 'cost' in user_message.lower()) and not any(term.lower() in response.lower() or term.lower() in user_message.lower() for term in pricing_terms.keys()):
        suggested_filters.append("pricing:Subscription")
    
    # Check for other useful filters
    if 'mobile' in response.lower() or 'app' in response.lower() or 'mobile' in user_message.lower():
        suggested_filters.append("mobile_app_available:Yes")
    
    if 'justice tech' in response.lower() or 'justice tech' in user_message.lower():
        suggested_filters.append("justice_tech_assn_member:Yes")
    
    return suggested_filters[:3]  # Limit to 3 suggested filters


def calculate_relevance(query, provider, embeddings=None):
    """Calculate relevance score between query and provider
    
    This is a simplified version. In a production environment, this would be handled
    by the vector database in the backend.
    
    Args:
        query: The search query
        provider: The provider data
        embeddings: Optional embeddings from the API (future use)
    
    Returns:
        A relevance score between 0 and 1
    """
    query_lower = query.lower()
    relevance_score = 0
    
    # Exact match boosting
    # Product name match (highest weight)
    if query_lower in provider['product_name'].lower():
        relevance_score += 0.6
    
    # Problem solved match
    if query_lower in provider['problem_that_product_solves'].lower():
        relevance_score += 0.5
    
    # Elevator pitch match
    if query_lower in provider['elevator_pitch'].lower():
        relevance_score += 0.4
    
    # Service area match
    if query_lower in provider['service_area'].lower():
        relevance_score += 0.3
    
    # Jurisdiction match
    jurisdictions = provider['jurisdiction'].split(', ') if isinstance(provider['jurisdiction'], str) else provider['jurisdiction']
    for jurisdiction in jurisdictions:
        if query_lower in jurisdiction.lower():
            relevance_score += 0.2
            break
    
    # Primary users match
    if query_lower in provider['primary_users'].lower():
        relevance_score += 0.2
    
    # Partial match relevance (if no exact matches were found)
    if relevance_score == 0:
        # Split query into words
        query_words = query_lower.split()
        
        # Check for partial matches in key fields
        key_fields = [
            ('product_name', 0.4),
            ('problem_that_product_solves', 0.3),
            ('elevator_pitch', 0.3),
            ('service_area', 0.2),
            ('primary_users', 0.1)
        ]
        
        # Count word matches in each field
        for field, weight in key_fields:
            field_text = provider[field].lower()
            match_count = sum(1 for word in query_words if word in field_text)
            if match_count > 0:
                # Calculate proportional match and add weighted score
                proportion = match_count / len(query_words)
                relevance_score += weight * proportion
    
    # If we had embeddings from the API, we could use them here
    # This would be a more advanced semantic search capability
    
    return min(relevance_score, 1.0)  # Cap at 1.0

if __name__ == '__main__':
    app.run(debug=True)