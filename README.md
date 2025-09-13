The Empathy Engine 🎙️
Give AI a human voice.
An advanced web app that generates expressive speech based on the emotion of your text.
🌟 Overview
The Empathy Engine is a full-stack web application designed to move beyond monotone, robotic speech. It analyzes text to determine its emotional content and then uses a multi-engine Text-to-Speech (TTS) system to generate audio that reflects the detected emotion.
✨ Core Features
🧠 Multi-Engine Emotion Analysis: Switch between VADER, TextBlob, and a Hugging Face DistilRoBERTa model for nuanced emotion detection.
🔊 Multi-Engine TTS Synthesis: Supports multiple TTS backends, from free options like gTTS to high-quality synthesis with Google Cloud TTS.
🔄 Intelligent Fallback System: If the primary TTS engine fails, the app automatically falls back to a secondary one for high availability.
🎭 Granular Emotional Mapping: Maps text to seven distinct emotional levels (e.g., Very Positive, Slightly Negative) for finer control over vocal output.
⚙️ Configurable via Environment: Easily switch emotion and TTS engines by setting environment variables—no code changes needed.
🌐 Modern Web Interface: Clean, responsive UI built with vanilla HTML, CSS, and JavaScript.
🩺 Health & Config Endpoints: Includes /health and /config API endpoints for easy monitoring and introspection.
🛠 Tech Stack
Category	Technology
Backend	Flask
Frontend	HTML, CSS, JavaScript
Emotion Analysis	VADER, TextBlob, Hugging Face Transformers
Speech Synthesis	gTTS, pyttsx3, Google Cloud TTS, macOS say
Deployment	Python venv, Environment Variables (.env)
🚀 Getting Started
1. Clone the Repository
git clone <your-github-repo-url>
cd <repository-folder-name>
2. Set Up the Environment
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
3. Install Dependencies
Create a requirements.txt file:
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
Install all packages:
pip install -r requirements.txt
4. Configure Your Engines
Create a .env file in the project root:
# --- OPTIONAL: Set your preferred engines ---
EMOTION_ENGINE=ensemble   # Options: vader, huggingface, textblob, ensemble
TTS_ENGINE=gtts           # Options: gtts, pyttsx3, google_cloud, macos_say

# To enable Google Cloud TTS, provide the path to your credentials file.
GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/google-credentials.json"
5. Run the Application
python app.py
The app will start locally (default: http://127.0.0.1:5001).
🎨 Design Philosophy
Emotion Analysis: The ensemble mode prioritizes Hugging Face for accuracy, but gracefully falls back to VADER for reliability.
Speech Synthesis: Abstracts the complexity of different TTS engines. For gTTS, different accents (via tld) simulate emotional variation—offering vocal variety at no extra cost.
This fallback-first approach makes the Empathy Engine both powerful for advanced setups and accessible for minimal ones.
🤝 Contributing
Contributions, issues, and feature requests are welcome!
Check the issues page.
📄 License
This project is licensed under the MIT License. See the LICENSE file for details.