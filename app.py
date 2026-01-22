import os
from flask import Flask, render_template, request, jsonify
from model import HousePriceModel

# App Configuration
app = Flask(__name__)
estimator = HousePriceModel()

def startup_validation():
    """Initializes the ML model or trains it if weights are absent."""
    if not estimator.load_model():
        print("Pre-trained model not detected. Commencing training...")
        from model import train_and_save_model
        train_and_save_model()
        estimator.load_model()

# Run initialization
startup_validation()

# Structural constraints for property data
DATA_BOUNDARIES = {
    'sq_ft': (500, 10000, 'Living Area (SqFt)'),
    'beds': (1, 10, 'Total Bedrooms'),
    'baths': (1, 8, 'Total Bathrooms'),
    'age': (0, 100, 'House Age'),
    'parking': (0, 4, 'Garage Capacity'),
    'rank': (1, 10, 'Neighborhood Rating')
}

@app.route('/')
def dashboard():
    """Serves the front-end valuation portal."""
    return render_template('index.html')

@app.route('/estimate-price', methods=['POST'])
def process_valuation_request():
    """Main endpoint for generating house price predictions."""
    try:
        req_payload = request.get_json() or {}

        # Extract and parse inputs with defaults
        inputs = {
            'square_feet': float(req_payload.get('square_feet', 0)),
            'bedrooms': int(req_payload.get('bedrooms', 0)),
            'bathrooms': float(req_payload.get('bathrooms', 0)),
            'age_years': int(req_payload.get('age_years', 0)),
            'garage_spaces': int(req_payload.get('garage_spaces', 0)),
            'location_score': int(req_payload.get('location_score', 0))
        }

        # Value Range Verification
        check_list = {
            'sq_ft': inputs['square_feet'],
            'beds': inputs['bedrooms'],
            'baths': inputs['bathrooms'],
            'age': inputs['age_years'],
            'parking': inputs['garage_spaces'],
            'rank': inputs['location_score']
        }

        for key, val in check_list.items():
            minimum, maximum, label = DATA_BOUNDARIES[key]
            if not (minimum <= val <= maximum):
                return jsonify({
                    'status': 'validation_error',
                    'reason': f"{label} is outside permissible range ({minimum}-{maximum})"
                }), 400

        # Perform Inference
        predicted_value = estimator.predict(**inputs)

        return jsonify({
            'success': True,
            'results': {
                'estimated_price': round(float(predicted_value), 2),
                'unit': 'USD',
                'meta': inputs
            }
        })

    except Exception as err:
        return jsonify({
            'success': False,
            'error_log': str(err)
        }), 500

if __name__ == '__main__':
    # Listen on environment-defined port or default to 8080
    listen_port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=listen_port, debug=False)