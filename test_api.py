"""
Test script for the Heart Disease Prediction API.
This script demonstrates how to use the API endpoints.
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def print_section(title):
    """Print a section header"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def test_health_check():
    """Test the health check endpoint"""
    print_section("Health Check")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

def test_root_endpoint():
    """Test the root endpoint"""
    print_section("Root Endpoint")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status Code: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

def test_prediction(description, data):
    """Test a prediction with given data"""
    print_section(f"Prediction: {description}")
    print("Input data:")
    print(json.dumps(data, indent=2))
    
    response = requests.post(f"{BASE_URL}/predict", json=data)
    print(f"\nStatus Code: {response.status_code}")
    print("Response:")
    print(json.dumps(response.json(), indent=2))

def main():
    """Run all tests"""
    print("\n" + "*"*60)
    print(" HEART DISEASE PREDICTION API - TEST SUITE")
    print("*"*60)
    
    # Test health check
    test_health_check()
    
    # Test root endpoint
    test_root_endpoint()
    
    # Test predictions
    test_prediction(
        "Low Risk - Young, healthy profile",
        {
            "gender": 1,
            "age": 25,
            "tc": 150.0,
            "hdl": 60.0,
            "smoke": 0,
            "bpm": 0,
            "diabetes": 0
        }
    )
    
    test_prediction(
        "Medium Risk - Middle-aged with some risk factors",
        {
            "gender": 0,
            "age": 54,
            "tc": 180.0,
            "hdl": 42.0,
            "smoke": 0,
            "bpm": 1,
            "diabetes": 0
        }
    )
    
    test_prediction(
        "High Risk - Older with multiple risk factors",
        {
            "gender": 0,
            "age": 75,
            "tc": 280.0,
            "hdl": 25.0,
            "smoke": 1,
            "bpm": 1,
            "diabetes": 1
        }
    )
    
    # Test validation
    print_section("Validation Test - Invalid Age")
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json={
                "gender": 0,
                "age": 150,  # Invalid: > 120
                "tc": 180.0,
                "hdl": 40.0,
                "smoke": 0,
                "bpm": 0,
                "diabetes": 0
            }
        )
        print(f"Status Code: {response.status_code}")
        print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "*"*60)
    print(" ALL TESTS COMPLETED")
    print("*"*60 + "\n")

if __name__ == "__main__":
    main()
