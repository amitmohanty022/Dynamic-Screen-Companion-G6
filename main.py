import os
import time
import threading
import pyttsx3
import google.generativeai as genai
from dotenv import load_dotenv
from flask import Flask, render_template, send_file
import io
import base64
from PIL import ImageGrab
import pytesseract
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain.prompts import MessagesPlaceholder, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.tools import Tool
from langchain_core.messages import SystemMessage

# Configure Tesseract path for Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load environment variables
load_dotenv()

# Initialize LangChain Agents
def create_agent_with_memory(agent_name, system_message):
    # Initialize the LLM
    llm = ChatOpenAI(
        temperature=0,
        model="gpt-3.5-turbo-1106",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Create conversation summary memory
    memory = ConversationSummaryMemory(
        llm=llm,
        memory_key="chat_history",
        return_messages=True
    )
    
    # Define tools for the agent
    tools = [
        Tool(
            name="screen_analysis",
            func=lambda x: "Screen analysis tool",
            description="Useful for analyzing screen content"
        ),
        Tool(
            name="text_extraction",
            func=lambda x: "Text extraction tool",
            description="Useful for extracting text from screenshots"
        )
    ]
    
    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_message),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Create the agent
    agent = create_openai_functions_agent(llm, tools, prompt)
    
    # Create the agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True
    )
    
    return agent_executor

# Create two agents with different purposes
screen_analysis_agent = create_agent_with_memory(
    "screen_analysis_agent",
    "You are a screen analysis assistant. Your role is to analyze screen content and provide insights."
)

text_processing_agent = create_agent_with_memory(
    "text_processing_agent",
    "You are a text processing assistant. Your role is to process and analyze extracted text from screenshots."
)

# Initialize Flask app
app = Flask(__name__, static_folder='static')

# Configure Google AI models
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel('gemma-3')

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
                
                # For Gemma 3
                prompt = f"""Analyze this text extracted from a screenshot: 
                
                {extracted_text}
                
                Please provide a brief description of what this content likely represents and any important information it contains."""
                
                response = model.generate_content(prompt)
                
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