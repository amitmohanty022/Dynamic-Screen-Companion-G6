import io
import os
import shutil
import threading
import time

import pyttsx3
from dotenv import load_dotenv
from flask import Flask, render_template, send_file
from google import genai
from PIL import ImageGrab

try:
    import pytesseract
except ImportError:  # OCR is optional; Gemini can read the screenshot directly
    pytesseract = None

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

if not API_KEY:
    raise SystemExit("GOOGLE_API_KEY is not set. Copy .env.example to .env and paste your Gemini API key.")

client = genai.Client(api_key=API_KEY)

# =============================================
# TESSERACT OCR CONFIGURATION (optional)
# =============================================

_WINDOWS_TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if pytesseract is not None:
    custom_path = os.getenv("TESSERACT_CMD")
    if custom_path:
        pytesseract.pytesseract.tesseract_cmd = custom_path
    elif os.name == "nt" and os.path.exists(_WINDOWS_TESSERACT):
        pytesseract.pytesseract.tesseract_cmd = _WINDOWS_TESSERACT

OCR_AVAILABLE = pytesseract is not None and (
    shutil.which(pytesseract.pytesseract.tesseract_cmd) is not None
    or os.path.exists(pytesseract.pytesseract.tesseract_cmd)
)


def configure_tesseract():
    """Configure Tesseract OCR with custom settings"""
    return r"--oem 3 --psm 6 -l eng"  # OEM 3: Default, PSM 6: Assume uniform block of text


def enhanced_ocr(image, config=None):
    """Extract text from a PIL image with Tesseract. Returns "" if OCR is unavailable."""
    if not OCR_AVAILABLE:
        return ""
    try:
        if config is None:
            config = configure_tesseract()
        if image.mode != "L":
            image = image.convert("L")
        return pytesseract.image_to_string(image, config=config).strip()
    except Exception as e:
        print(f"Enhanced OCR Error: {str(e)}")
        return ""


def batch_ocr(images, config=None):
    """Process multiple images with OCR"""
    return [enhanced_ocr(image, config) for image in images]


# Initialize Flask app
app = Flask(__name__, static_folder="static", template_folder="templates")


class ScreenAssistant:
    def __init__(self):
        self.is_running = False
        self.current_screenshot = None
        self.conversation_history = []

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
            greeting = "Welcome! I'm your screen companion. Starting screen analysis now."
            self.speak(greeting)
            self.conversation_history.append({"role": "assistant", "content": greeting})

            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()

            # Wait briefly for first screenshot
            time.sleep(1)
            self.analyze_screen()
        except Exception as e:
            print(f"Initial capture error: {e}")

    def stop_capture(self):
        if not self.is_running:
            return True

        self.is_running = False
        if hasattr(self, "capture_thread"):
            self.capture_thread.join(timeout=1.0)

        self.speak("Thank you for using Screen Companion. Have a great day!")
        self.conversation_history = []
        self.current_screenshot = None
        return True

    def _capture_loop(self):
        while self.is_running:
            try:
                screenshot = self.take_screenshot()
                if screenshot:
                    self.current_screenshot = screenshot
            except Exception as e:
                print(f"Capture Error: {str(e)}")
            time.sleep(2.0)

    def analyze_screen(self):
        if not self.current_screenshot:
            return
        try:
            extracted_text = self.extract_text()
            prompt = (
                "Analyze this screenshot of the user's screen. "
                "Provide a brief description of what this content likely represents and any important "
                "information it contains. Keep it short and easy to read aloud."
            )
            if extracted_text:
                prompt += f"\n\nText extracted from the screen with OCR:\n{extracted_text}"

            # Gemini is multimodal, so the screenshot itself is sent along with the prompt.
            screenshot = self.current_screenshot.copy()
            screenshot.thumbnail((1600, 1600))
            response = client.models.generate_content(model=MODEL, contents=[prompt, screenshot])

            if response.text:
                analysis_text = response.text
                print("Analysis:", analysis_text)
                self.conversation_history.append({"role": "assistant", "content": analysis_text})
                self.speak(analysis_text)
        except Exception as e:
            print(f"Analysis Error: {str(e)}")

    def extract_text(self):
        """Extract text from the current screenshot using OCR (if Tesseract is installed)."""
        try:
            if self.current_screenshot:
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
            engine.setProperty("rate", 150)
            engine.setProperty("volume", 0.9)
            print("Speaking:", text)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
            del engine
        except Exception as e:
            print(f"Speech Error: {str(e)}")


screen_assistant = ScreenAssistant()


# Routes
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/start")
def start_capture():
    screen_assistant.start_capture()
    return "Screen capture started"


@app.route("/stop")
def stop_capture():
    if screen_assistant.stop_capture():
        return "Stream stopped"
    return "Error stopping stream", 500


@app.route("/get_preview")
def get_preview():
    if screen_assistant.current_screenshot:
        img_io = io.BytesIO()
        screen_assistant.current_screenshot.save(img_io, "PNG")
        img_io.seek(0)
        return send_file(img_io, mimetype="image/png")
    return "", 404


if __name__ == "__main__":
    print(f"OCR (Tesseract) available: {OCR_AVAILABLE}")
    print("Open http://127.0.0.1:5000 in your browser")
    app.run(host="127.0.0.1", port=5000, debug=False)
