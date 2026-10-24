# Dynamic Screen Companion

## Overview

Dynamic Screen Companion is a Python-based application designed to be your intelligent assistant by continuously analyzing your screen content in real-time. It captures screenshots, extracts text using Optical Character Recognition (OCR), and leverages the power of language models (currently Google's Gemini) to provide insightful analysis and feedback. It also features text-to-speech capabilities to communicate findings audibly.

This project utilizes a Flask web interface to allow users to start and stop the screen analysis and view a live preview of the captured screen.

## 🔗 Live demo

This app captures **your own screen** and speaks through your speakers, so it runs locally rather than on a hosted website. Follow the installation steps below, then open http://127.0.0.1:5000/. An extended version (FastAPI, demo, tests, Docker) lives in [kirodata/dynamic-screen-companion](https://github.com/amitmohanty022/kirodata/tree/main/dynamic-screen-companion).

## Features

* **Real-time Screen Capture:** Continuously captures screenshots in the background.
* **Optical Character Recognition (OCR):** Extracts text from the captured screenshots using Tesseract OCR with enhanced configuration for better accuracy.
* **Intelligent Screen Analysis:** Utilizes Google's Gemini language model to analyze the extracted text and provide meaningful descriptions and insights about the screen content.
* **Text-to-Speech Feedback:** Communicates the analysis results audibly using `pyttsx3`.
* **Web Interface (Flask):** Provides a simple web interface to:
    * Start and stop the screen capture and analysis.
    * View a live preview of the currently captured screen.
* **Modular Design:** The codebase is structured with a `ScreenAssistant` class to encapsulate the core logic, making it easier to understand and extend.
* **Environment Variable Management:** Uses `.env` files to securely manage API keys.

## Technologies Used

* Python 3
* Flask
* Pillow (PIL)
* pytesseract
* pyttsx3
* google-genai (Google Gen AI SDK, model `gemini-3.8-flash` by default)
* python-dotenv

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/amitmohanty022/Dynamic-Screen-Companion-G6.git
    cd Dynamic-Screen-Companion-G6
    ```

2.  **Install the required Python libraries:**
    ```bash
    pip install -r requirements.txt
    ```
    *(It's recommended to create and activate a virtual environment before installing dependencies.)*

3.  **Install Tesseract OCR:**
    * **Windows:** Download and install Tesseract OCR from [https://tesseract-ocr.github.io/tessdoc/Installation.html](https://tesseract-ocr.github.io/tessdoc/Installation.html). The default install path `C:\Program Files\Tesseract-OCR\tesseract.exe` is detected automatically. If you installed it elsewhere, set `TESSERACT_CMD` in your `.env` file.
    * **Linux (Debian/Ubuntu):**
        ```bash
        sudo apt update
        sudo apt install tesseract-ocr libtesseract-dev
        ```
    * **macOS:**
        ```bash
        brew install tesseract
        ```
    * Refer to the official Tesseract documentation for installation instructions on other operating systems.

4.  **Set up Environment Variables:**
    * Create a `.env` file in the root directory of the project.
    * Add your Google Gemini API key to the `.env` file:
        ```
        GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY
        # optional
        GEMINI_MODEL=gemini-3.8-flash
        TESSERACT_CMD=/path/to/tesseract
        ```
        *(Get a Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey).)*

## Usage

1.  **Run the Flask application:**
    ```bash
    python main.py
    ```

2.  **Open your web browser and navigate to `http://127.0.0.1:5000/`**.

3.  **Use the buttons on the web interface:**
    * **"Start Capture"**: Initiates the continuous screen capture and analysis process. The assistant will start analyzing your screen content and speaking the results.
    * **"Stop Capture"**: Halts the screen capture and analysis.

4.  The "Current Preview" section will display a near real-time view of the captured screen.

## Future Enhancements

* **Integration of LangChain Agents:** Add LangChain agents for more specialized and context-aware screen analysis, such as identifying UI elements, reading tables, or summarizing articles.
* **More Sophisticated Analysis:** Implement more complex prompting and potentially fine-tune language models for better accuracy and more detailed insights.
* **User Configuration:** Allow users to configure parameters such as the screen capture frequency, analysis verbosity, and preferred voice for text-to-speech.
* **Selective Region Capture:** Enable users to select a specific region of the screen for analysis instead of the entire screen.
* **Web Interface Improvements:** Enhance the web interface to display the conversation history, provide more controls, and offer different visualization options.
* **Cross-Platform Compatibility:** Ensure smoother setup and operation across different operating systems.
* **Saving Analysis History:** Implement a feature to save the history of screen analyses.

## Contributing

Contributions to the Dynamic Screen Companion project are welcome! If you have ideas for improvements, bug fixes, or new features, feel free to open an issue or submit a pull request. Please follow standard GitHub practices for contributing.

## License

No license file has been added yet.

## Acknowledgements

* This project utilizes the power of Google's Gemini API for intelligent analysis.
* The OCR functionality is powered by the Tesseract OCR engine.
* The text-to-speech functionality is provided by the `pyttsx3` library.
* The web interface is built using the Flask microframework.

---

**Created by [Amit Kumar Mohanty](https://github.com/amitmohanty022)**
