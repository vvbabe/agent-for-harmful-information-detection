#!/usr/bin/env python3
"""
 - Gradio
"""

import sys
import subprocess
import time

def test_dependencies():
    """"""
    print(" ...")
    
    required_modules = {
        'gradio': 'Gradio Web',
        'pandas': '',
        'numpy': '', 
        'plotly': ''
    }
    
    failed_modules = []
    
    for module, description in required_modules.items():
        try:
            __import__(module)
            print(f" {module} - {description}")
        except ImportError:
            print(f" {module} - {description} ()")
            failed_modules.append(module)
    
    if failed_modules:
        print(f"\n  : {', '.join(failed_modules)}")
        print(": pip install -r requirements.txt")
        return False
    
    print(" ")
    return True

def test_frontend_import():
    """"""
    print("\n ...")
    
    try:
        from frontend_app import create_interface, ThreatDetectionFrontend
        print(" ")
        
        # 
        app = ThreatDetectionFrontend()
        print(" ")
        
        # 
        demo = create_interface()
        print(" Gradio")
        
        return True
        
    except Exception as e:
        print(f" : {str(e)}")
        return False

def test_model_api():
    """API"""
    print("\n API...")
    
    try:
        from model_api import ModelAPI, predict_threat
        
        # API
        api = ModelAPI()
        print(" API")
        
        # 
        result = predict_threat("test input", use_rag=True)
        print(" ")
        print(f"   : {result.get('prediction', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f" API: {str(e)}")
        return False

def main():
    """"""
    print(" ...\n")
    
    test_results = []
    
    # 
    test_results.append(test_dependencies())
    
    # 
    test_results.append(test_frontend_import())
    
    # API
    test_results.append(test_model_api())
    
    # 
    print("\n" + "="*50)
    print(" ")
    print("="*50)
    
    if all(test_results):
        print(" ")
        print("\n :")
        print("   1. : python run_app.py")
        print("   2. : http://localhost:7860")
        print("   3. ")
        return 0
    else:
        print(" ")
        print("\n :")
        print("   1. : pip install -r requirements.txt")
        print("   2. Python >= 3.8")
        print("   3. ")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
