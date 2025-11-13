"""
test_complete_pipeline.py - Test complete fact-checking pipeline
"""
import json
from app.features.factcheck_pipe import FactCheckPipe as fcp
from app.models.factcheck import FactCheckResponse
from pydantic import ValidationError


def test_pipeline_json_structure():
    """Test that FactCheckPipe returns correct JSON for frontend"""
    print("="*70)
    print("FactCheck Pipeline JSON Structure Test")
    print("="*70)

    text = """
    The Eiffel Tower was built in 1889 for the World's Fair in Paris.
    It stands 324 meters tall and was designed by Gustave Eiffel.
    The tower is located in London, England and weighs 50,000 tons.
    Albert Einstein was born in 1879 in Germany.
    """

    print("\n" + "="*70)
    print("RUNNING PIPELINE")
    print("="*70)
    print(f"Input text: {text.strip()[:100]}...\n")
    
    try:
        # Run the pipeline
        response = fcp.run(text)
        
        print("✓ Pipeline executed successfully\n")
        
        # Print raw response for inspection
        print("="*70)
        print("RAW PIPELINE OUTPUT")
        print("="*70)
        print(json.dumps(response, indent=2, default=str))
        
        # Validate against Pydantic model
        print("\n" + "="*70)
        print("PYDANTIC MODEL VALIDATION")
        print("="*70)
        
        try:
            validated_response = FactCheckResponse(**response)
            print("✓ Response structure matches FactCheckResponse model")
            
            # Print validated structure
            print("\n" + "="*70)
            print("VALIDATED JSON (What frontend will receive)")
            print("="*70)
            print(json.dumps(validated_response.model_dump(), indent=2, default=str))
            
            # Check specific fields your frontend needs
            print("\n" + "="*70)
            print("FRONTEND CONTRACT VALIDATION")
            print("="*70)
            
            response_dict = validated_response.model_dump()
            
            # Add your specific frontend requirements here
            required_fields = []  # e.g., ["claims", "overall_verdict", "confidence"]
            
            if required_fields:
                for field in required_fields:
                    if field in response_dict:
                        print(f"✓ Required field '{field}' present")
                    else:
                        print(f"✗ Required field '{field}' MISSING")
            else:
                print("ℹ No specific frontend fields defined to check")
                print(f"Available fields: {list(response_dict.keys())}")
            
            print("\n" + "="*70)
            print("TEST RESULT: ✓ PASSED")
            print("="*70)
            print("The pipeline returns valid JSON for your frontend!")
            
            return True
            
        except ValidationError as e:
            print("✗ Response structure does NOT match FactCheckResponse model")
            print("\nValidation Errors:")
            print(e)
            print("\n" + "="*70)
            print("TEST RESULT: ✗ FAILED")
            print("="*70)
            return False
            
    except Exception as e:
        print(f"\n✗ Pipeline execution failed with error:")
        print(f"   {type(e).__name__}: {str(e)}")
        print("\n" + "="*70)
        print("TEST RESULT: ✗ FAILED")
        print("="*70)
        return False


def test_pipeline_with_various_inputs():
    """Test pipeline with different input types"""
    print("\n\n" + "="*70)
    print("TESTING WITH VARIOUS INPUTS")
    print("="*70)
    
    test_cases = [
        ("Single claim", "The Earth is round."),
        ("Multiple claims", "Water boils at 100°C. Ice is cold."),
        ("Empty string", ""),
        ("Very long text", "Lorem ipsum. " * 100),
    ]
    
    for name, text in test_cases:
        print(f"\n{name}:")
        try:
            response = fcp.run(text)
            validated = FactCheckResponse(**response)
            print(f"  ✓ Valid JSON returned")
        except ValidationError as e:
            print(f"  ✗ Invalid JSON structure: {e}")
        except Exception as e:
            print(f"  ✗ Pipeline error: {e}")


if __name__ == "__main__":
    # Run main test
    success = test_pipeline_json_structure()
    
    # Run additional tests
    test_pipeline_with_various_inputs()
    
    # Exit with appropriate code
    exit(0 if success else 1)