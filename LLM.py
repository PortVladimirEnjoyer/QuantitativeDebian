#disclaimer(note) : ai was used for this code 


import ollama
from threading import Thread
import sys
import time
import subprocess
import requests
import logging
from packaging import version

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ollama_chat.log')
    ]
)
logger = logging.getLogger(__name__)

# Configuration
OLLAMA_HOST = "http://localhost:11434"
SERVER_START_TIMEOUT = 15  # Increased timeout
POLL_INTERVAL = 1
MODEL_NAME = "wizard-math"

def is_ollama_installed():
    """Check if Ollama is installed and available in PATH."""
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"Ollama version: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.error(f"Ollama not found in PATH: {e}")
        return False

def is_ollama_running():
    """Check if Ollama server is responsive."""
    try:
        response = requests.get(
            f"{OLLAMA_HOST}/api/tags",
            timeout=3
        )
        if response.status_code == 200:
            logger.debug("Ollama server is responsive")
            return True
        logger.warning(f"Unexpected API response: HTTP {response.status_code}")
        return False
    except requests.RequestException as e:
        logger.debug(f"Server connection failed: {str(e)}")
        return False

def start_ollama_server():
    """Start Ollama server with comprehensive error handling."""
    try:
        logger.info("Starting Ollama server...")
        process = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # Check for immediate failure
        time.sleep(1)
        if process.poll() is not None:
            stderr = process.stderr.read()
            logger.error(f"Server failed to start. Error:\n{stderr}")
            return None

        logger.info("Ollama server process started")
        return process

    except Exception as e:
        logger.error(f"Failed to start server process: {e}")
        return None

def wait_for_server_start():
    """Wait for server to become responsive with timeout."""
    logger.info("Waiting for server to start...")
    start_time = time.time()
    
    while time.time() - start_time < SERVER_START_TIMEOUT:
        if is_ollama_running():
            logger.info("Server is now responsive")
            return True
        time.sleep(POLL_INTERVAL)
    
    logger.error(f"Server did not become responsive within {SERVER_START_TIMEOUT} seconds")
    return False

def ensure_model_exists():
    """Ensure the specified model exists with robust error handling."""
    try:
        # Get model list with error handling
        try:
            response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
            response.raise_for_status()
            models_data = response.json()
            
            if not isinstance(models_data, dict) or 'models' not in models_data:
                raise ValueError("Invalid response format from Ollama API")
                
            models = models_data['models']
        except Exception as e:
            logger.error(f"Failed to get model list: {e}")
            models = []

        # Check for model existence
        model_exists = any(
            isinstance(m, dict) and m.get('name') == MODEL_NAME
            for m in models
        )

        if not model_exists:
            logger.info(f"Model '{MODEL_NAME}' not found. Downloading...")
            try:
                pull_process = subprocess.Popen(
                    ["ollama", "pull", MODEL_NAME],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                # Stream download progress
                while True:
                    output = pull_process.stdout.readline()
                    if output == '' and pull_process.poll() is not None:
                        break
                    if output:
                        logger.info(output.strip())
                
                if pull_process.returncode != 0:
                    error = pull_process.stderr.read()
                    logger.error(f"Download failed: {error}")
                    return False
                
                logger.info(f"Model '{MODEL_NAME}' successfully downloaded")
                return True
                
            except Exception as e:
                logger.error(f"Model download failed: {e}")
                return False
        else:
            logger.info(f"Model '{MODEL_NAME}' is available")
            return True

    except Exception as e:
        logger.error(f"Model verification failed: {e}")
        return False

def stream_response(prompt):
    """Stream the model's response with error handling."""
    try:
        stream = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )
        print("\nAssistant:\n")
        for chunk in stream:
            print(chunk["message"]["content"], end="", flush=True)
        print("\n")
    except Exception as e:
        print(f"\nError generating response: {str(e)}\n")

def chat_loop():
    """Run interactive chat session."""
    print(f"\n=== {MODEL_NAME} Chat ===")
    print("Type 'exit' or 'quit' to end the session\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ["exit", "quit"]:
                print("\nGoodbye!\n")
                break
            
            Thread(target=stream_response, args=(user_input,)).start()
            
        except KeyboardInterrupt:
            print("\nSession ended by user")
            break
        except Exception as e:
            print(f"\nInput error: {str(e)}\n")

def main():
    """Main application flow."""
    print(
        """
             

 .----------------.  .----------------.  .----------------.  .----------------.  .----------------.  .----------------.   .----------------.  .----------------.  .----------------.  .----------------. 
| .--------------. || .--------------. || .--------------. || .--------------. || .--------------. || .--------------. | | .--------------. || .--------------. || .--------------. || .--------------. |
| | _____  _____ | || |     _____    | || |   ________   | || |      __      | || |  _______     | || |  ________    | | | | ____    ____ | || |      __      | || |  _________   | || |  ____  ____  | |
| ||_   _||_   _|| || |    |_   _|   | || |  |  __   _|  | || |     /  \     | || | |_   __ \    | || | |_   ___ `.  | | | ||_   \  /   _|| || |     /  \     | || | |  _   _  |  | || | |_   ||   _| | |
| |  | | /\ | |  | || |      | |     | || |  |_/  / /    | || |    / /\ \    | || |   | |__) |   | || |   | |   `. \ | | | |  |   \/   |  | || |    / /\ \    | || | |_/ | | \_|  | || |   | |__| |   | |
| |  | |/  \| |  | || |      | |     | || |     .'.' _   | || |   / ____ \   | || |   |  __ /    | || |   | |    | | | | | |  | |\  /| |  | || |   / ____ \   | || |     | |      | || |   |  __  |   | |
| |  |   /\   |  | || |     _| |_    | || |   _/ /__/ |  | || | _/ /    \ \_ | || |  _| |  \ \_  | || |  _| |___.' / | | | | _| |_\/_| |_ | || | _/ /    \ \_ | || |    _| |_     | || |  _| |  | |_  | |
| |  |__/  \__|  | || |    |_____|   | || |  |________|  | || ||____|  |____|| || | |____| |___| | || | |________.'  | | | ||_____||_____|| || ||____|  |____|| || |   |_____|    | || | |____||____| | |
| |              | || |              | || |              | || |              | || |              | || |              | | | |              | || |              | || |              | || |              | |
| '--------------' || '--------------' || '--------------' || '--------------' || '--------------' || '--------------' | | '--------------' || '--------------' || '--------------' || '--------------' |
 '----------------'  '----------------'  '----------------'  '----------------'  '----------------'  '----------------'   '----------------'  '----------------'  '----------------'  '----------------' 


                                                                                                                      
                                                                                                                          
                                                                                                                      
       """
    )
    try:
        # Verify Ollama installation
        if not is_ollama_installed():
            logger.error("Ollama not installed. Download from https://ollama.ai")
            sys.exit(1)

        # Server management
        server_process = None
        if not is_ollama_running():
            server_process = start_ollama_server()
            if not server_process or not wait_for_server_start():
                logger.error("Failed to start Ollama server")
                if server_process:
                    server_process.terminate()
                sys.exit(1)

        # Model verification
        if not ensure_model_exists():
            logger.error(f"Failed to verify/download model '{MODEL_NAME}'")
            sys.exit(1)

        # Start chat session
        chat_loop()

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
    finally:
        if server_process:
            logger.info("Terminating server process")
            server_process.terminate()

if __name__ == "__main__":
    main()
