# Dynamic Screen Companion

## Overview

Dynamic Screen Companion is a Python-based application designed to be your intelligent assistant by continuously analyzing your screen content in real-time. It captures screenshots, extracts text using Optical Character Recognition (OCR), and leverages the power of language models (currently Google's Gemini) to provide insightful analysis and feedback. It also features text-to-speech capabilities to communicate findings audibly.

This project utilizes a Flask web interface to allow users to start and stop the screen analysis and view a live preview of the captured screen. While it currently uses Gemini for analysis, it also lays the groundwork for integrating LangChain agents with memory and tools, potentially allowing for more sophisticated and specialized screen analysis tasks in the future.

## Features

* **Real-time Screen Capture:** Continuously captures screenshots in the background.
* **Optical Character Recognition (OCR):** Extracts text from the captured screenshots using Tesseract OCR with enhanced configuration for better accuracy.
* **Intelligent Screen Analysis:** Utilizes Google's Gemini language model to analyze the extracted text and provide meaningful descriptions and insights about the screen content.
* **Text-to-Speech Feedback:** Communicates the analysis results audibly using `pyttsx3`.
* **Web Interface (Flask):** Provides a simple web interface to:
    * Start and stop the screen capture and analysis.
    * View a live preview of the currently captured screen.
* **Modular Design:** The codebase is structured with a `ScreenAssistant` class to encapsulate the core logic, making it easier to understand and extend.
* **Potential for LangChain Integration:** Includes initial setup for LangChain agents with memory and tools, paving the way for more advanced functionalities in future updates.
* **Environment Variable Management:** Uses `.env` files to securely manage API keys.

## Technologies Used

* Python 3
* Flask
* Pillow (PIL)
* pytesseract
* pyttsx3
* google-generativeai
* python-dotenv
* LangChain (optional, for future agent integration)
* OpenAI (optional, for future LangChain agent integration)

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/Dynamic-Screen-Companion.git](https://github.com/YOUR_USERNAME/Dynamic-Screen-Companion.git)
    cd Dynamic-Screen-Companion
    ```
    *(Replace `YOUR_USERNAME` with your GitHub username and `Dynamic-Screen-Companion` with the actual repository name if different)*

2.  **Install the required Python libraries:**
    ```bash
    pip install -r requirements.txt
    ```
    *(It's recommended to create and activate a virtual environment before installing dependencies.)*

3.  **Install Tesseract OCR:**
    * **Windows:** Download and install Tesseract OCR from [https://tesseract-ocr.github.io/tessdoc/Installation.html](https://tesseract-ocr.github.io/tessdoc/Installation.html). After installation, make sure to update the `pytesseract.pytesseract.tesseract_cmd` variable in the script (`main.py`) to the correct path of your `tesseract.exe` file (e.g., `r'C:\Program Files\Tesseract-OCR\tesseract.exe'`).
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
        # Add your OpenAI API key if you plan to explore LangChain agents
        # OPENAI_API_KEY=YOUR_OPENAI_API_KEY
        ```
        *(Replace `YOUR_GOOGLE_API_KEY` with your actual Google Gemini API key. You'll need to obtain this from the Google Cloud AI Platform.)*

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

* **Integration of LangChain Agents:** Fully implement and utilize the LangChain agents for more specialized and context-aware screen analysis. This could involve creating tools for specific tasks like identifying UI elements, reading data from tables, or summarizing articles.
* **More Sophisticated Analysis:** Implement more complex prompting and potentially fine-tune language models for better accuracy and more detailed insights.
* **User Configuration:** Allow users to configure parameters such as the screen capture frequency, analysis verbosity, and preferred voice for text-to-speech.
* **Selective Region Capture:** Enable users to select a specific region of the screen for analysis instead of the entire screen.
* **Web Interface Improvements:** Enhance the web interface to display the conversation history, provide more controls, and offer different visualization options.
* **Cross-Platform Compatibility:** Ensure smoother setup and operation across different operating systems.
* **Saving Analysis History:** Implement a feature to save the history of screen analyses.

## Contributing

Contributions to the Dynamic Screen Companion project are welcome! If you have ideas for improvements, bug fixes, or new features, feel free to open an issue or submit a pull request. Please follow standard GitHub practices for contributing.

## License

[Specify your project's license here, e.g., MIT License]

## Acknowledgements

* This project utilizes the power of Google's Gemini API for intelligent analysis.
* The OCR functionality is powered by the Tesseract OCR engine.
* The text-to-speech functionality is provided by the `pyttsx3` library.
* The web interface is built using the Flask microframework.
* The LangChain library provides a framework for building intelligent agents.

---

**Created by [Your Name/GitHub Username]**
