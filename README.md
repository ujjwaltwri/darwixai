The Empathy Engine 🎤
The Empathy Engine is an advanced, full-stack web application designed to give AI a more human voice. It analyzes input text to determine its emotional content and then uses a multi-engine Text-to-Speech (TTS) system to generate audio that reflects the detected emotion. This project moves beyond monotonic, robotic speech to create a more expressive and engaging user experience.

This application was built to fulfill the "AI Beyond Words" hackathon challenge, demonstrating a sophisticated approach to both sentiment analysis and dynamic speech synthesis.

✨ Features
Multi-Engine Emotion Analysis: Dynamically switch between VADER, TextBlob, and a powerful Hugging Face DistilRoBERTa model for nuanced emotion detection. An ensemble mode uses the best available engine.

Multi-Engine TTS Synthesis: Supports various TTS backends, including free options like gTTS and pyttsx3, the native macOS say command, and high-quality cloud synthesis with Google Cloud TTS.

Intelligent Fallback System: If a primary TTS engine fails, the application automatically falls back to a secondary engine, ensuring high availability.

Granular Emotional Mapping: Instead of a simple positive/negative/neutral model, the engine maps text to seven distinct emotional levels (e.g., "Very Positive", "Slightly Negative") for finer control over the vocal output.

Configurable via Environment: Easily switch the active emotion and TTS engines by setting environment variables—no code changes needed.

Modern Web Interface: A clean, responsive, and intuitive UI built with vanilla HTML, CSS, and JavaScript for a seamless user experience.

Health & Config Endpoints: Includes /health and /config API endpoints for easy monitoring and introspection of the application's status and available engines.

🛠️ Setup and Installation
Follow these steps to set up and run the project locally.

1. Clone the Repository

Bash
git clone <your-github-repo-url>
cd <repository-folder-name>
2. Create and Activate a Virtual Environment

It's highly recommended to use a virtual environment to manage dependencies.

Bash
# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate
3. Install Dependencies

This project requires several packages. You can install them all by creating a requirements.txt file with the content below and running pip install.

First, create a file named requirements.txt:

flask
python-dotenv
vaderSentiment
gtts
pyttsx3
# Optional but recommended
transformers
torch
textblob
# For Google Cloud TTS
google-cloud-texttospeech
Now, install everything with a single command:

Bash
pip install -r requirements.txt
Note: The Hugging Face transformers library can be large. It's optional but required for the best emotion analysis.

4. Configure Environment Variables

Create a file named .env in the root of the project directory. This file will store your configuration and API keys.

Code snippet
# --- REQUIRED ---
# No keys are strictly required to run, as the app falls back to free engines.

# --- OPTIONAL ---
# Set the default engines. Options are listed in the file.
# EMOTION_ENGINE=ensemble  # (vader, huggingface, textblob, ensemble)
# TTS_ENGINE=gtts          # (gtts, pyttsx3, google_cloud, macos_say)

# To enable Google Cloud TTS, uncomment the following line and
# point it to your JSON credentials file.
# GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/google-credentials.json"
5. Run the Application

Once the dependencies are installed and the .env file is configured, run the Flask application:

Bash
python app.py
The application will start on a free port (usually 5000) and display the URL in the terminal. Open that URL in your web browser to use the Empathy Engine.

🎨 Design Choices
Emotion Analysis

The core of this project is its ability to understand emotion with nuance.

Layered Approach: I implemented three different sentiment analysis libraries. VADER is excellent for social media text and provides a reliable baseline. TextBlob offers a simple polarity score. The Hugging Face DistilRoBERTa model is the most powerful, trained specifically to classify text into emotions like joy, sadness, and anger, providing much richer context than a simple sentiment score.

Ensemble Default: The default ensemble mode uses the best available engine (prioritizing Hugging Face) and falls back gracefully, making the application robust and powerful out of the box.

Granular Mapping: The continuous scores from VADER and TextBlob are mapped to a 7-point emotional scale, allowing for more detailed vocal modulation than a simple positive/negative binary.

Speech Synthesis

The goal was to create a flexible and resilient TTS system.

Engine Abstraction: All TTS engines are wrapped in a single TTSEngine class. This makes it easy to add new engines in the future and provides a consistent interface for the main application.

Emotional Voice Mapping: For each TTS engine, I designed a specific mapping from the 7 emotional levels to appropriate vocal parameters.

For gTTS, I used different top-level domains (tld) to access different accents (e.g., Australian for "Very Positive", UK for "Positive"), which provides a creative and effective way to change the voice's character.

For pyttsx3, I directly modulated the rate and volume and attempted to switch between available system voices.

For Google Cloud TTS, I selected specific high-quality WaveNet voices that are known for their expressive range.

Smart Fallbacks: The system is designed to never fail. If the primary, high-quality engine (like Google Cloud) is unavailable or encounters an error, the application automatically tries a configured fallback, and then any other available free engine, ensuring that the user always gets an audio output.