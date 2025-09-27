#!/usr/bin/env python3
"""
Modern Chatbot Backend API
Connects to trained LLM models for intelligent conversations
"""

import os
import json
import torch
import asyncio
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

# Import our trained models
from model.transformer import TinyTransformerLM
from tokenizer.spm_tokenizer import SPMTokenizer

app = FastAPI(
    title="AI Chatbot API",
    description="Modern chatbot powered by trained LLM models",
    version="1.0.0"
)

# CORS middleware for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variables
model = None
tokenizer = None
device = None

class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = []

class ChatResponse(BaseModel):
    response: str
    status: str = "success"

class ModelManager:
    """Manages the trained model loading and inference"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        
    def load_model(self, checkpoint_path: str = "checkpoints/final_model.pt"):
        """Load the trained model and tokenizer"""
        try:
            print(f"Loading model from {checkpoint_path}...")
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            config = checkpoint['config']
            
            # Initialize tokenizer
            self.tokenizer = SPMTokenizer("tokenizer/spm.model")
            config['vocab_size'] = self.tokenizer.VOCAB_SIZE
            
            # Initialize model
            self.model = TinyTransformerLM(**config)
            self.model.load_state_dict(checkpoint['model_state'])
            
            # Set device
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
                
            self.model.to(self.device)
            self.model.eval()
            
            print(f"Model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def generate_response(self, prompt: str, max_tokens: int = 20) -> str:
        """Generate response using the trained model"""
        try:
            if not self.model or not self.tokenizer:
                return "Model not loaded. Please try again."
            
            # Encode input
            input_ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=True)
            print(f"Input prompt: {prompt}")
            print(f"Input IDs: {input_ids}")
            
            # Simple generation loop
            generated_ids = input_ids.copy()
            self.model.eval()
            
            with torch.no_grad():
                for step in range(max_tokens):
                    # Get current sequence
                    current_tensor = torch.tensor([generated_ids], dtype=torch.long).to(self.device)
                    
                    # Get logits
                    logits = self.model(current_tensor)
                    next_token_logits = logits[0, -1, :] / 0.7  # Temperature
                    
                    # Remove EOS token from consideration
                    next_token_logits[self.tokenizer.EOS] = float('-inf')
                    
                    # Get top 10 tokens and sample from them
                    top_k = 10
                    top_logits, top_indices = torch.topk(next_token_logits, top_k)
                    
                    # Sample from top tokens
                    probs = torch.softmax(top_logits, dim=-1)
                    selected_idx = torch.multinomial(probs, num_samples=1).item()
                    next_token = top_indices[selected_idx].item()
                    
                    print(f"Step {step}: Generated token {next_token}")
                    
                    # Add to sequence
                    generated_ids.append(next_token)
                    
                    # Stop if we hit a reasonable length
                    if len(generated_ids) > len(input_ids) + 15:
                        print("Reached max length, stopping generation")
                        break
            
            # Decode response
            response_ids = generated_ids[len(input_ids):]
            print(f"Response IDs: {response_ids}")
            response = self.tokenizer.decode(response_ids)
            print(f"Raw response: '{response}'")
            
            # Clean up response
            response = response.strip()
            if response.endswith('</s>'):
                response = response[:-4]
            
            # Clean up response further
            response = response.replace('⁇', '').replace('  ', ' ').strip()
            
            print(f"Final response: '{response}'")
            return response if response else "I'm not sure how to respond to that."
            
        except Exception as e:
            print(f"Error generating response: {e}")
            import traceback
            traceback.print_exc()
            return "I'm not sure how to respond to that."

# Initialize model manager
model_manager = ModelManager()

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    print("Starting AI Chatbot API...")
    success = model_manager.load_model()
    if not success:
        print("Warning: Model loading failed. API will run with fallback responses.")

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the chatbot frontend"""
    try:
        with open("chatbot.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Chatbot Frontend Not Found</h1>", status_code=404)

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint"""
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Build context from history
        context = ""
        for msg in request.history[-5:]:  # Last 5 messages for context
            role = "User" if msg.get("role") == "user" else "Assistant"
            context += f"{role}: {msg.get('content', '')}\n"
        
        # Create prompt
        prompt = f"{context}User: {request.message}\nAssistant:"
        
        # Generate response
        response = model_manager.generate_response(prompt)
        
        return ChatResponse(response=response)
        
    except Exception as e:
        print(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_manager.model is not None,
        "device": str(model_manager.device) if model_manager.device else "unknown"
    }

@app.get("/api/model/info")
async def model_info():
    """Get model information"""
    if not model_manager.model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": "TinyTransformerLM",
        "vocab_size": model_manager.tokenizer.VOCAB_SIZE if model_manager.tokenizer else 0,
        "device": str(model_manager.device) if model_manager.device else "unknown",
        "checkpoint": "final_model.pt"
    }

if __name__ == "__main__":
    print("Starting AI Chatbot Server...")
    print("Frontend will be available at: http://localhost:8000")
    print("API documentation at: http://localhost:8000/docs")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
