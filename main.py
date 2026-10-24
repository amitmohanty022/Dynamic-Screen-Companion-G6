import os
import io
import shutil
import threading
import time

import pyttsx3
import pytesseract
from dotenv import load_dotenv
from flask import Flask, render_template, send_file
from google import genai
from PIL import ImageGrab

# Load environment variables
load_dotenv()

# Configure the Tesseract path. Set TESSERACT_CMD in .env if tesseract is not on your PATH
# (on Windows the default install location is used as a fallback).
_tesseract_cmd = os.getenv("TESSERACT_CMD") or shutil.which("tesseract")
if not _tesseract_cmd and os.name == "nt":
    _tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if _tesseract_cmd:
    pytesseract.pytesseract.tesseract_cmd = _tesseract_cmd

# Initialize Flask app (templates/ holds index.html, static/css/ holds style.css)
app = Flask(__name__, static_folder="static", template_folder="templates")

# Configure the Google Gemini client (google-genai SDK)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

class ScreenAssistant:
    def __init__(self):
        self.is_running = False
        self.current_screenshot = None
        self.conversation_history = []
        self.engine = None  # Don't initialize here

    def take_screenshot(self):
        try:
            return ImageGrab.grab()
        except Exception as e:
            print(f"Screenshot Error: {str(e)}")
            return None

    def start_capture(self):
        if self.is_running:  # Prevent multiple starts
            return
            
        self.is_running = True
        try:
            # Greeting message
            greeting = "Welcome! I'm your screen companion. Starting screen analysis now."
            self.speak(greeting)
            self.conversation_history.append({
                "role": "assistant",
                "content": greeting
            })

            # Start capture thread first
            self.capture_thread = threading.Thread(target=self._capture_loop)
            self.capture_thread.daemon = True
            self.capture_thread.start()

            # Wait briefly for first screenshot
            time.sleep(1)
            self.analyze_screen()
            
        except Exception as e:
            print(f"Initial capture error: {e}")

    def stop_capture(self):
        if not self.is_running:  # Prevent multiple stops
            return True
            
        self.is_running = False
        
        # Wait for capture thread to end
        if hasattr(self, 'capture_thread'):
            self.capture_thread.join(timeout=1.0)
        
        # Farewell message
        farewell = "Thank you for using Screen Companion. Have a great day!"
        self.speak(farewell)
        
        self.conversation_history = []
        self.current_screenshot = None
        return True

    def _capture_loop(self):
        while self.is_running:
            try:
                screenshot = self.take_screenshot()
                if screenshot:
                    self.current_screenshot = screenshot
                    time.sleep(0.1)  # Brief delay between captures
            except Exception as e:
                print(f"Capture Error: {str(e)}")
            time.sleep(2.0)  

    def analyze_screen(self):
        if self.current_screenshot:
            try:
                # Extract text using OCR
                extracted_text = self.extract_text()
                
                prompt = f"""Analyze this text extracted from a screenshot: 
                
                {extracted_text}
                
                Please provide a brief description of what this content likely represents and any important information it contains."""
                
                response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
                
                if response.text:
                    analysis_text = response.text
                    print("Analysis:", analysis_text)
                    self.conversation_history.append({
                        "role": "assistant", 
                        "content": analysis_text
                    })
                    self.speak(analysis_text)
                    
            except Exception as e:
                print(f"Analysis Error: {str(e)}")

    def extract_text(self):
        """Extract text from the current screenshot using OCR."""
        try:
            if self.current_screenshot:
                # Use enhanced OCR instead of basic OCR
                text = enhanced_ocr(self.current_screenshot)
                print("Extracted text:", text)
                return text
            return ""
        except Exception as e:
            print(f"OCR Error: {str(e)}")
            return ""

    def speak(self, text):
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 0.9)
            print("Speaking:", text)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
            del engine  # Explicitly delete the engine
        except Exception as e:
            print(f"Speech Error: {str(e)}")

# =============================================
# TESSERACT OCR CONFIGURATION AND FUNCTIONS
# =============================================

def configure_tesseract():
    """Configure Tesseract OCR with custom settings"""
    custom_config = r'--oem 3 --psm 6 -l eng+equ'  # OEM 3: Default, PSM 6: Assume uniform block of text
    return custom_config

def enhanced_ocr(image, config=None):
    """
    Enhanced OCR function with error handling and preprocessing
    
    Args:
        image: PIL Image object
        config: Optional custom Tesseract configuration
        
    Returns:
        str: Extracted text
    """
    try:
        if config is None:
            config = configure_tesseract()
            
        # Convert image to grayscale for better OCR
        if image.mode != 'L':
            image = image.convert('L')
            
        # Perform OCR with custom configuration
        text = pytesseract.image_to_string(image, config=config)
        
        # Clean up the text
        text = text.strip()
        
        return text
    except Exception as e:
        print(f"Enhanced OCR Error: {str(e)}")
        return ""

def batch_ocr(images, config=None):
    """
    Process multiple images with OCR
    
    Args:
        images: List of PIL Image objects
        config: Optional custom Tesseract configuration
        
    Returns:
        list: List of extracted texts
    """
    results = []
    for image in images:
        text = enhanced_ocr(image, config)
        results.append(text)
    return results

# Initialize screen assistant
screen_assistant = ScreenAssistant()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start')
def start_capture():
    screen_assistant.start_capture()
    return 'Screen capture started'

@app.route('/stop')
def stop_capture():
    if screen_assistant.stop_capture():
        return 'Stream stopped'
    return 'Error stopping stream', 500

@app.route('/get_preview')
def get_preview():
    if screen_assistant.current_screenshot:
        img_io = io.BytesIO()
        screen_assistant.current_screenshot.save(img_io, 'PNG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png')
    return '', 404

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False) 