import subprocess
import sys
import os
import time

def install_dependencies():
    print("Installing dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        return False
    return True

def check_environment():
    print("Checking environment...")
    
    if not os.path.exists(".env"):
        print("Warning: .env file not found")
        print("Please create a .env file with your API keys")
        print("You can copy .env.example and fill in your keys")
    
    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key or openai_key == 'your-api-key-here':
        print("Warning: OPENAI_API_KEY not set or using placeholder")
        print("The system will use a fallback mechanism for LLM calls")
    
    return True

def run_tests():
    print("Running system tests...")
    try:
        result = subprocess.run([sys.executable, "test_system.py"], 
                              capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("Tests passed!")
            return True
        else:
            print(f"Tests failed: {result.stdout}")
            print(f"Errors: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("Tests timed out")
        return False
    except Exception as e:
        print(f"Error running tests: {e}")
        return False

def start_server():
    print("Starting server...")
    print("Server will be available at: http://localhost:8000")
    print("API documentation at: http://localhost:8000/docs")
    print("Press Ctrl+C to stop the server")
    
    try:
        subprocess.run([sys.executable, "start_server.py"])
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error starting server: {e}")

def main():
    print("=== LLM Query-Retrieval System Deployment ===")
    
    if not install_dependencies():
        return
    
    if not check_environment():
        return
    
    choice = input("\nRun tests first? (y/n): ").lower().strip()
    if choice == 'y':
        if not run_tests():
            choice = input("Tests failed. Continue anyway? (y/n): ").lower().strip()
            if choice != 'y':
                return
    
    start_server()

if __name__ == "__main__":
    main()