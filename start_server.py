import uvicorn
import os
import sys
from pathlib import Path

if __name__ == "__main__":
    root_dir = Path(__file__).parent
    sys.path.insert(0, str(root_dir))
    
    if not os.path.exists(".env"):
        print("Warning: .env file not found. Using environment variables or defaults.")
        print("Please create a .env file based on .env.example")
    
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"Starting server on {host}:{port}")
    print("Available endpoints:")
    print(f"  - POST http://{host}:{port}/hackrx/run")
    print(f"  - GET  http://{host}:{port}/health")
    print(f"  - GET  http://{host}:{port}/docs (API documentation)")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
        access_log=True,
        workers=1
    )