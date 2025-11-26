"""
Tools to simulate exactly what the frontend receives from your API.
"""

import json
import requests
from app.features.factcheck_pipe import FactCheckPipe
from app.models.factcheck import FactCheckResponse


# ============================================================================
# Method 1: Direct Pipeline Simulation (No Server Required)
# ============================================================================

def simulate_frontend_request(text: str, config: dict = None):
    """
    Simulate exactly what happens when frontend sends a request.
    This bypasses the API and calls the pipeline directly.
    """
    print("=" * 80)
    print("SIMULATING FRONTEND REQUEST")
    print("=" * 80)
    print(f"\nInput Text: '{text}'")
    print(f"Config: {config}")
    
    # Initialize pipeline (optionally with custom config)
    pipe = FactCheckPipe()
    
    # Apply config if provided
    if config:
        for key, value in config.items():
            if hasattr(pipe.cfg, key.upper()):
                setattr(pipe.cfg, key.upper(), value)
                print(f"  Set {key.upper()} = {value}")
    
    print("\n" + "-" * 80)
    print("PROCESSING...")
    print("-" * 80)
    
    try:
        # This is exactly what your API endpoint does
        response = pipe.process(text)
        
        # Convert to dict (what gets serialized to JSON)
        response_dict = response.dict()
        
        # Pretty print the response
        print("\n" + "=" * 80)
        print("RESPONSE (JSON that frontend receives):")
        print("=" * 80)
        print(json.dumps(response_dict, indent=2, default=str))
        
        # Summary stats
        print("\n" + "=" * 80)
        print("SUMMARY:")
        print("=" * 80)
        print(f"Number of claims detected: {len(response_dict['claims'])}")
        print(f"Summary stats: {response_dict['summary']}")
        
        if len(response_dict['claims']) == 0:
            print("\n⚠️  WARNING: No claims detected!")
            print("Debugging info:")
            print(f"  - Cleaned text: {pipe.cleaned_text}")
            print(f"  - Sentences: {pipe.sentences}")
            print(f"  - Claim threshold: {pipe.detector.claim_threshold}")
            
            # Check each sentence
            if pipe.sentences:
                print("\n  Sentence-by-sentence analysis:")
                for i, sent in enumerate(pipe.sentences):
                    is_claim, conf, ctype = pipe.detector.is_factual_claim(sent)
                    print(f"    {i+1}. '{sent}'")
                    print(f"       -> is_claim={is_claim}, confidence={conf:.3f}, type={ctype}")
        
        return response_dict
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


# ============================================================================
# Method 2: Test Against Running Server
# ============================================================================

def test_live_api(text: str, base_url: str = "http://localhost:8000"):
    """
    Test against your actual running FastAPI server.
    Use this to ensure API serialization works correctly.
    """
    print("=" * 80)
    print("TESTING LIVE API")
    print("=" * 80)
    print(f"\nBase URL: {base_url}")
    print(f"Input Text: '{text}'")
    
    try:
        # Test the factcheck endpoint
        print("\n1. Testing /api/factcheck endpoint...")
        response = requests.post(
            f"{base_url}/api/factcheck",
            json={"text": text},
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        print(f"   Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("\n" + "=" * 80)
            print("RESPONSE:")
            print("=" * 80)
            print(json.dumps(data, indent=2))
            
            print("\n" + "=" * 80)
            print("SUMMARY:")
            print("=" * 80)
            print(f"Claims detected: {len(data.get('claims', []))}")
            
            return data
        else:
            print(f"   Error: {response.text}")
            return {"error": response.text}
            
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Cannot connect to server.")
        print(f"   Make sure your API is running at {base_url}")
        return {"error": "Connection failed"}
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        return {"error": str(e)}


# ============================================================================
# Method 3: Compare Frontend vs Backend Serialization
# ============================================================================

def compare_serialization(text: str):
    """
    Compare what the pipeline produces vs what JSON serialization creates.
    Catches issues with Pydantic model serialization.
    """
    print("=" * 80)
    print("SERIALIZATION COMPARISON")
    print("=" * 80)
    
    pipe = FactCheckPipe()
    
    # Get the Pydantic response
    print("\n1. Processing through pipeline...")
    response = pipe.process(text)
    
    print(f"   Type: {type(response)}")
    print(f"   Claims count: {len(response.claims)}")
    
    # Test different serialization methods
    print("\n2. Testing serialization methods...")
    
    # Method 1: .dict()
    try:
        dict_result = response.dict()
        print("   ✓ .dict() works")
        print(f"     Claims in dict: {len(dict_result['claims'])}")
    except Exception as e:
        print(f"   ✗ .dict() failed: {e}")
        dict_result = None
    
    # Method 2: .json()
    try:
        json_str = response.json()
        json_result = json.loads(json_str)
        print("   ✓ .json() works")
        print(f"     Claims in JSON: {len(json_result['claims'])}")
    except Exception as e:
        print(f"   ✗ .json() failed: {e}")
        json_result = None
    
    # Method 3: json.dumps with default=str
    try:
        json_str = json.dumps(dict_result, indent=2, default=str)
        print("   ✓ json.dumps() works")
    except Exception as e:
        print(f"   ✗ json.dumps() failed: {e}")
    
    # Check if results match
    if dict_result and json_result:
        print("\n3. Comparing results...")
        if dict_result == json_result:
            print("   ✓ Both serialization methods produce identical results")
        else:
            print("   ⚠️  Serialization methods produce different results!")
    
    return dict_result


# ============================================================================
# Method 4: Batch Testing
# ============================================================================

def batch_test_claims(test_cases: list):
    """
    Test multiple claims at once to see patterns.
    """
    print("=" * 80)
    print("BATCH TESTING")
    print("=" * 80)
    
    pipe = FactCheckPipe()
    results = []
    
    for i, test_text in enumerate(test_cases, 1):
        print(f"\n{i}. Testing: '{test_text}'")
        
        try:
            response = pipe.process(test_text)
            response_dict = response.dict()
            
            num_claims = len(response_dict['claims'])
            print(f"   Claims detected: {num_claims}")
            
            if num_claims > 0:
                for claim in response_dict['claims']:
                    print(f"     - {claim['claim']['text']}")
            
            results.append({
                "input": test_text,
                "claims_detected": num_claims,
                "success": True
            })
            
        except Exception as e:
            print(f"   ❌ ERROR: {str(e)}")
            results.append({
                "input": test_text,
                "claims_detected": 0,
                "success": False,
                "error": str(e)
            })
    
    # Summary
    print("\n" + "=" * 80)
    print("BATCH TEST SUMMARY")
    print("=" * 80)
    successful = sum(1 for r in results if r['success'])
    with_claims = sum(1 for r in results if r['claims_detected'] > 0)
    
    print(f"Total tests: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Detected claims: {with_claims}")
    print(f"Failed: {len(results) - successful}")
    
    return results


# ============================================================================
# Method 5: Interactive REPL
# ============================================================================

def interactive_test():
    """
    Interactive testing session - type claims and see results immediately.
    """
    print("=" * 80)
    print("INTERACTIVE TESTING MODE")
    print("=" * 80)
    print("Type claims to test (or 'quit' to exit)")
    print("-" * 80)
    
    pipe = FactCheckPipe()
    
    while True:
        text = input("\nEnter text to fact-check: ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("Exiting...")
            break
        
        if not text:
            continue
        
        try:
            response = pipe.process(text)
            response_dict = response.dict()
            
            print(f"\n✓ Claims detected: {len(response_dict['claims'])}")
            
            if len(response_dict['claims']) > 0:
                for i, claim_verification in enumerate(response_dict['claims'], 1):
                    claim = claim_verification['claim']
                    verdict = claim_verification.get('verdict', 'UNKNOWN')
                    confidence = claim_verification.get('confidence', 0.0)
                    
                    print(f"\n  Claim {i}: {claim['text']}")
                    print(f"    Verdict: {verdict}")
                    print(f"    Confidence: {confidence:.2f}")
            else:
                print("  ⚠️  No claims detected")
                print(f"  Threshold: {pipe.detector.claim_threshold}")
                
        except Exception as e:
            print(f"\n❌ ERROR: {str(e)}")


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    # Test 1: Simulate the problematic input
    print("\n\n")
    print("#" * 80)
    print("# TEST 1: Your problematic input")
    print("#" * 80)
    
    simulate_frontend_request(
        "the sky is blue and the sky is red and the sky is black."
    )
    
    # Test 2: Test against live API (if running)
    print("\n\n")
    print("#" * 80)
    print("# TEST 2: Live API test")
    print("#" * 80)
    
    # Uncomment if your server is running:
    # test_live_api("the sky is blue and the sky is red and the sky is black.")
    
    # Test 3: Batch test various inputs
    print("\n\n")
    print("#" * 80)
    print("# TEST 3: Batch testing")
    print("#" * 80)
    
    test_cases = [
        "the sky is blue and the sky is red and the sky is black.",
        "The sky is blue.",
        "Paris is the capital of France.",
        "The Eiffel Tower was built in 1889.",
        "I think the weather is nice today.",
        "What is the capital of France?",
    ]
    
    batch_test_claims(test_cases)
    
    # Test 4: Check serialization
    print("\n\n")
    print("#" * 80)
    print("# TEST 4: Serialization check")
    print("#" * 80)
    
    compare_serialization("Paris is the capital of France.")
    
    # Test 5: Interactive mode (uncomment to use)
    # print("\n\n")
    # print("#" * 80)
    # print("# TEST 5: Interactive mode")
    # print("#" * 80)
    # interactive_test()