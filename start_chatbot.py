#!/usr/bin/env python3
"""
AI Chatbot Launcher
Starts the complete chatbot system with frontend and backend
"""

import os
import sys
import subprocess
import webbrowser
import time
import signal
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    try:
        import fastapi
        import uvicorn
        import torch
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("Please install requirements: pip install -r backend/requirements.txt")
        return False

def start_backend():
    """Start the FastAPI backend server"""
    print("🚀 Starting AI Chatbot Backend...")
    
    # Change to project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    # Set Python path
    env = os.environ.copy()
    env['PYTHONPATH'] = str(project_dir)
    
    # Start the server
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "backend.app:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ], env=env, check=True)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down chatbot...")
    except Exception as e:
        print(f"❌ Error starting backend: {e}")

def open_browser():
    """Open browser to chatbot"""
    time.sleep(3)  # Wait for server to start
    try:
        webbrowser.open("http://localhost:8000")
        print("🌐 Opening chatbot in browser...")
    except Exception as e:
        print(f"Could not open browser: {e}")
        print("Please manually open: http://localhost:8000")

def main():
    """Main launcher function"""
    print("🤖 AI Chatbot System Launcher")
    print("=" * 40)
    
    # Check requirements
    if not check_requirements():
        return
    
    # Check if model files exist
    model_path = Path("checkpoints/final_model.pt")
    if not model_path.exists():
        print("❌ Model file not found!")
        print("Please make sure you have trained the model first.")
        print("Run the training scripts to create checkpoints/final_model.pt")
        return
    
    print("✅ Model files found")
    print("🚀 Starting chatbot system...")
    print("\n📱 Chatbot will be available at: http://localhost:8000")
    print("📚 API documentation at: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop the chatbot")
    print("=" * 40)
    
    # Start backend (this will block)
    start_backend()

if __name__ == "__main__":
    main()
