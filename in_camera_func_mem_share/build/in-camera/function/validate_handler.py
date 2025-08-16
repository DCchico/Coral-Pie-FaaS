#!/usr/bin/env python3
"""
Validation script for handler.py
Checks for potential issues before deployment
"""

import sys
import os
import importlib
import inspect
import ast

def check_imports():
    """Check if all required imports are available"""
    print("🔍 Checking imports...")
    
    required_imports = [
        'cv2', 'json', 'logging', 'traceback', 'time', 'os', 
        'jsonpickle', 'base64', 'numpy', 'pickle', 'sys',
        'multiprocessing.shared_memory', 'struct', 'posix_ipc', 'requests'
    ]
    
    missing_imports = []
    for module in required_imports:
        try:
            importlib.import_module(module)
            print(f"  ✅ {module}")
        except ImportError as e:
            print(f"  ❌ {module}: {e}")
            missing_imports.append(module)
    
    if missing_imports:
        print(f"\n⚠️ Missing imports: {missing_imports}")
        return False
    else:
        print("✅ All imports available")
        return True

def check_function_signatures():
    """Check if function signatures are correct"""
    print("\n🔍 Checking function signatures...")
    
    try:
        # Import the handler module
        sys.path.insert(0, os.path.dirname(__file__))
        import handler
        
        # Check key functions
        functions_to_check = [
            ('parse_remote_info', 1),
            ('store_frame_state', 6),
            ('get_previous_frame_state', 7),
            ('handle', 2)
        ]
        
        for func_name, expected_params in functions_to_check:
            if hasattr(handler, func_name):
                func = getattr(handler, func_name)
                sig = inspect.signature(func)
                param_count = len(sig.parameters)
                print(f"  ✅ {func_name}: {param_count} parameters")
                
                if param_count != expected_params:
                    print(f"    ⚠️ Expected {expected_params} parameters, got {param_count}")
            else:
                print(f"  ❌ {func_name}: Function not found")
                return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error checking functions: {e}")
        return False

def check_syntax():
    """Check if the handler file has valid Python syntax"""
    print("\n🔍 Checking syntax...")
    
    try:
        with open('handler.py', 'r') as f:
            content = f.read()
        
        # Parse the AST
        ast.parse(content)
        print("  ✅ Syntax is valid")
        return True
        
    except SyntaxError as e:
        print(f"  ❌ Syntax error: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Error checking syntax: {e}")
        return False

def check_global_variables():
    """Check if required global variables are defined"""
    print("\n🔍 Checking global variables...")
    
    try:
        with open('handler.py', 'r') as f:
            content = f.read()
        
        # Simple check for key global variables
        required_globals = [
            'tgraph', 'vt', 'STATE_PROXY_TIMEOUT', 'camera_states'
        ]
        
        for var in required_globals:
            if var in content:
                print(f"  ✅ {var}")
            else:
                print(f"  ❌ {var}: Not found")
                return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error checking globals: {e}")
        return False

def check_remote_parsing():
    """Test the remote parsing function"""
    print("\n🔍 Testing remote parsing...")
    
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        import handler
        
        test_cases = [
            ("remote:192.168.1.10:8080", (True, "192.168.1.10", 8080)),
            ("remote:localhost:9090", (True, "localhost", 9090)),
            ("shared_memory", (False, None, None)),
            ("", (False, None, None)),
            ("remote:invalid", (False, None, None)),
            ("remote:192.168.1.10:invalid", (False, None, None))
        ]
        
        for test_input, expected in test_cases:
            result = handler.parse_remote_info(test_input)
            if result == expected:
                print(f"  ✅ '{test_input}' -> {result}")
            else:
                print(f"  ❌ '{test_input}' -> {result} (expected {expected})")
                return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error testing remote parsing: {e}")
        return False

def main():
    """Run all validation checks"""
    print("🚀 Starting Handler Validation...")
    print("=" * 50)
    
    checks = [
        check_syntax,
        check_imports,
        check_global_variables,
        check_function_signatures,
        check_remote_parsing
    ]
    
    results = []
    for check in checks:
        try:
            result = check()
            results.append(result)
        except Exception as e:
            print(f"  ❌ Check failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("📊 Validation Results:")
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print("✅ All checks passed! Handler is ready for testing.")
        print("\n🎯 Next steps:")
        print("1. Run: python handler.py (for local testing)")
        print("2. Build and deploy to your FaaS cluster")
        print("3. Test with your Go StateProxy")
    else:
        print(f"❌ {total - passed} check(s) failed. Please fix issues before deployment.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 