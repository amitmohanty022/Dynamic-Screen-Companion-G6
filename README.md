# Dynamic Screen Companion

An assistant that watches your screen, works out what you are looking at with Google Gemini, and **reads its analysis out loud**. A small web page lets you start and stop it and shows a live preview of the captured screen.

## Features

- **Real-time screen capture** every 2 seconds in the background.
- **AI screen analysis**: the screenshot is sent to Gemini (`gemini-3.8-flash` by default), which describes what is on screen.
- **Optional OCR**: if Tesseract is installed, the extracted text is sent too, which helps with small text.
- **Text-to-speech** with `pyttsx3`, which works offline.
- **Web interface (Flask)** with Start/Stop buttons and a live preview.

## How to run (step by step)

You need **Python 3.10 or newer** ([download](https://www.python.org/downloads/)) on **Windows or macOS**. Linux works too with an X11 desktop.

**1. Download the project**

```bash
git clone https://github.com/amitmohanty022/Dynamic-Screen-Companion-G6.git
cd Dynamic-Screen-Companion-G6
```

**2. Create a virtual environment and activate it**

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

**3. Install the requirements**

```bash
pip install -r requirements.txt
```

Linux only: also run `sudo apt install espeak` for the voice.

**4. Add your Gemini API key**

- Get a free key at https://aistudio.google.com/apikey
- Copy `.env.example` to a new file named `.env`
  - Windows: `copy .env.example .env`
  - macOS / Linux: `cp .env.example .env`
- Open `.env` and paste your key after `GOOGLE_API_KEY=`

**5. (Optional) Install Tesseract OCR for better text reading**

The app works without it. If you want it:

- Windows: install from https://github.com/UB-Mannheim/tesseract/wiki (default folder `C:\Program Files\Tesseract-OCR` is detected automatically)
- macOS: `brew install tesseract`
- Ubuntu/Debian: `sudo apt install tesseract-ocr`

If you installed it somewhere else, set `TESSERACT_CMD=` in `.env` to the full path of `tesseract`.

**6. Start the app**

```bash
python main.py
```

Open http://127.0.0.1:5000 in your browser.

**7. Use it**

- **Start Capture**: begins capturing, analyzes the screen, and speaks the result.
- **Stop Capture**: stops capturing.
- The preview area shows the latest screenshot.

macOS: the first time, allow **Screen Recording** for your Terminal app in *System Settings → Privacy & Security*, then restart the terminal.

## Settings

| Variable in `.env` | Default | What it does |
|---|---|---|
| `GOOGLE_API_KEY` | (empty, required) | Gemini key |
| `GEMINI_MODEL` | `gemini-3.8-flash` | Gemini model |
| `TESSERACT_CMD` | auto-detected | Path to the Tesseract program (optional) |

## Troubleshooting

| Problem | Fix |
|---|---|
| "GOOGLE_API_KEY is not set" | Make sure the file is named exactly `.env` (not `.env.txt`) and is in the project folder. |
| Black or empty preview on macOS | Give Screen Recording permission (see step 7). |
| No voice | Windows/macOS: check your volume. Linux: `sudo apt install espeak`. |
| `OCR (Tesseract) available: False` | Optional. Install Tesseract (step 5) if you want OCR. |

## Acknowledgements

Google Gemini API, Tesseract OCR, `pyttsx3`, Flask.

**Created by Amit Kumar Mohanty ([@amitmohanty022](https://github.com/amitmohanty022))**
