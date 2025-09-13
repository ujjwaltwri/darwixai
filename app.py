import os
import uuid
import json
import sys
import subprocess
from flask import Flask, request, jsonify, render_template, url_for
from dotenv import load_dotenv

# Emotion Analysis Imports
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
try:
    from transformers import pipeline
    HF_AVAILABLE = True
    print("Hugging Face transformers loaded successfully")
except ImportError:
    HF_AVAILABLE = False
    print("Hugging Face transformers not available. Install with: pip install transformers torch")

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
    print("TextBlob loaded successfully")
except ImportError:
    TEXTBLOB_AVAILABLE = False
    print("TextBlob not available. Install with: pip install textblob")

# TTS Imports
try:
    import pyttsx3
    # Test if pyttsx3 actually works
    test_engine = pyttsx3.init()
    test_engine.stop()
    PYTTSX3_AVAILABLE = True
    print("pyttsx3 available and working")
except Exception as e:
    PYTTSX3_AVAILABLE = False
    print(f"pyttsx3 not available: {e}")

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
    print("gTTS loaded successfully")
except ImportError:
    GTTS_AVAILABLE = False
    print("gTTS not available. Install with: pip install gtts")

try:
    from google.cloud import texttospeech
    GOOGLE_CLOUD_AVAILABLE = True
    print("Google Cloud TTS available")
except ImportError:
    GOOGLE_CLOUD_AVAILABLE = False
    print("Google Cloud TTS not available")

# --- Load Environment Variables ---
load_dotenv()

# --- Configuration ---
class Config:
    # Emotion Analysis Engine Options: 'vader', 'huggingface', 'textblob', 'ensemble'
    EMOTION_ENGINE = os.getenv('EMOTION_ENGINE', 'ensemble')
    
    # TTS Engine Options: 'gtts', 'pyttsx3', 'google_cloud', 'macos_say'
    TTS_ENGINE = os.getenv('TTS_ENGINE', 'gtts')
    FALLBACK_TTS = os.getenv('FALLBACK_TTS', 'pyttsx3' if PYTTSX3_AVAILABLE else 'macos_say')
    
    # API Keys
    GOOGLE_CLOUD_KEY_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    
    # Text limits
    MAX_TEXT_LENGTH = int(os.getenv('MAX_TEXT_LENGTH', '1000'))

# --- Initialize Clients ---
def initialize_clients():
    clients = {}
    
    # Google Cloud TTS
    if GOOGLE_CLOUD_AVAILABLE and Config.GOOGLE_CLOUD_KEY_PATH:
        try:
            clients['google_cloud'] = texttospeech.TextToSpeechClient()
            print("Google Cloud TTS client initialized")
        except Exception as e:
            print(f"Google Cloud TTS initialization failed: {e}")
    
    # Hugging Face Pipeline
    if HF_AVAILABLE:
        try:
            clients['emotion_classifier'] = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                device=-1  # Use CPU
            )
            print("Hugging Face emotion classifier initialized")
        except Exception as e:
            print(f"Could not load Hugging Face model: {e}")
    
    return clients

clients = initialize_clients()

# --- Initialize Flask App ---
app = Flask(__name__)

# --- Enhanced Emotion Analysis Engine ---
class EmotionAnalyzer:
    def __init__(self):
        self.vader_analyzer = SentimentIntensityAnalyzer()
    
    def analyze_vader(self, text):
        """VADER sentiment analysis"""
        sentiment = self.vader_analyzer.polarity_scores(text)
        compound = sentiment['compound']
        
        if compound >= 0.6:
            emotion = "Very Positive"
        elif compound >= 0.2:
            emotion = "Positive"
        elif compound >= 0.05:
            emotion = "Slightly Positive"
        elif compound <= -0.6:
            emotion = "Very Negative"
        elif compound <= -0.2:
            emotion = "Negative"
        elif compound <= -0.05:
            emotion = "Slightly Negative"
        else:
            emotion = "Neutral"
        
        return {
            'engine': 'VADER',
            'emotion': emotion,
            'confidence': abs(compound),
            'scores': {
                'compound': round(compound, 3),
                'positive': round(sentiment['pos'], 3),
                'negative': round(sentiment['neg'], 3),
                'neutral': round(sentiment['neu'], 3)
            }
        }
    
    def analyze_huggingface(self, text):
        """Hugging Face emotion classification"""
        if not HF_AVAILABLE or 'emotion_classifier' not in clients:
            return None
        
        try:
            results = clients['emotion_classifier'](text)
            top_emotion = results[0]
            
            # Map HF emotions to our categories
            emotion_mapping = {
                'joy': 'Very Positive',
                'surprise': 'Positive',
                'neutral': 'Neutral',
                'sadness': 'Negative',
                'anger': 'Very Negative',
                'fear': 'Very Negative',
                'disgust': 'Negative'
            }
            
            emotion = emotion_mapping.get(top_emotion['label'].lower(), 'Neutral')
            
            return {
                'engine': 'Hugging Face',
                'emotion': emotion,
                'confidence': round(top_emotion['score'], 3),
                'raw_emotion': top_emotion['label'],
                'all_scores': results
            }
        except Exception as e:
            print(f"Hugging Face analysis failed: {e}")
            return None
    
    def analyze_textblob(self, text):
        """TextBlob sentiment analysis"""
        if not TEXTBLOB_AVAILABLE:
            return None
        
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            if polarity >= 0.5:
                emotion = "Very Positive"
            elif polarity >= 0.1:
                emotion = "Positive"
            elif polarity >= 0.05:
                emotion = "Slightly Positive"
            elif polarity <= -0.5:
                emotion = "Very Negative"
            elif polarity <= -0.1:
                emotion = "Negative"
            elif polarity <= -0.05:
                emotion = "Slightly Negative"
            else:
                emotion = "Neutral"
            
            return {
                'engine': 'TextBlob',
                'emotion': emotion,
                'confidence': abs(polarity),
                'scores': {
                    'polarity': round(polarity, 3),
                    'subjectivity': round(subjectivity, 3)
                }
            }
        except Exception as e:
            print(f"TextBlob analysis failed: {e}")
            return None
    
    def analyze(self, text):
        """Main analysis method"""
        results = {}
        
        if Config.EMOTION_ENGINE == 'vader':
            results['primary'] = self.analyze_vader(text)
        elif Config.EMOTION_ENGINE == 'huggingface':
            hf_result = self.analyze_huggingface(text)
            results['primary'] = hf_result if hf_result else self.analyze_vader(text)
        elif Config.EMOTION_ENGINE == 'textblob':
            tb_result = self.analyze_textblob(text)
            results['primary'] = tb_result if tb_result else self.analyze_vader(text)
        elif Config.EMOTION_ENGINE == 'ensemble':
            # Run all available engines
            results['vader'] = self.analyze_vader(text)
            if HF_AVAILABLE:
                results['huggingface'] = self.analyze_huggingface(text)
            if TEXTBLOB_AVAILABLE:
                results['textblob'] = self.analyze_textblob(text)
            
            # Use Hugging Face as primary if available, otherwise VADER
            if 'huggingface' in results and results['huggingface']:
                results['primary'] = results['huggingface']
            else:
                results['primary'] = results['vader']
        
        return results

# --- Multi-Engine TTS System ---
class TTSEngine:
    def __init__(self):
        self.output_dir = os.path.join('static', 'audio')
        os.makedirs(self.output_dir, exist_ok=True)
    
    def get_voice_for_emotion(self, emotion, engine='gtts'):
        """Get appropriate voice based on emotion and TTS engine"""
        voice_mappings = {
            'gtts': {
                'Very Positive': {'tld': 'com.au', 'slow': False},  # Australian - upbeat
                'Positive': {'tld': 'co.uk', 'slow': False},        # British - pleasant
                'Slightly Positive': {'tld': 'ca', 'slow': False},  # Canadian - friendly
                'Neutral': {'tld': 'com', 'slow': False},           # US - standard
                'Slightly Negative': {'tld': 'co.in', 'slow': False}, # Indian - thoughtful
                'Negative': {'tld': 'co.za', 'slow': True},         # South African - slower
                'Very Negative': {'tld': 'com', 'slow': True}       # US - slow and somber
            },
            'google_cloud': {
                'Very Positive': ('en-US-Wavenet-H', 'FEMALE'),
                'Positive': ('en-US-Wavenet-C', 'FEMALE'),
                'Slightly Positive': ('en-US-Wavenet-E', 'MALE'),
                'Neutral': ('en-US-Wavenet-D', 'MALE'),
                'Slightly Negative': ('en-US-Wavenet-B', 'MALE'),
                'Negative': ('en-US-Wavenet-A', 'MALE'),
                'Very Negative': ('en-US-Wavenet-I', 'MALE')
            },
            'pyttsx3': {
                'Very Positive': {'rate': 200, 'volume': 1.0},
                'Positive': {'rate': 180, 'volume': 0.9},
                'Slightly Positive': {'rate': 170, 'volume': 0.8},
                'Neutral': {'rate': 160, 'volume': 0.7},
                'Slightly Negative': {'rate': 140, 'volume': 0.6},
                'Negative': {'rate': 130, 'volume': 0.5},
                'Very Negative': {'rate': 120, 'volume': 0.4}
            },
            'macos_say': {
                'Very Positive': {'voice': 'Samantha', 'rate': 200},
                'Positive': {'voice': 'Kathy', 'rate': 180},
                'Slightly Positive': {'voice': 'Alex', 'rate': 170},
                'Neutral': {'voice': 'Alex', 'rate': 160},
                'Slightly Negative': {'voice': 'Tom', 'rate': 140},
                'Negative': {'voice': 'Ralph', 'rate': 130},
                'Very Negative': {'voice': 'Fred', 'rate': 120}
            }
        }
        
        return voice_mappings.get(engine, {}).get(emotion, voice_mappings[engine].get('Neutral'))
    
    def synthesize_gtts(self, text, emotion, filename):
        """Google Text-to-Speech (gTTS)"""
        if not GTTS_AVAILABLE:
            return None
        
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            voice_settings = self.get_voice_for_emotion(emotion, 'gtts')
            tld = voice_settings['tld']
            slow = voice_settings['slow']
            
            tts = gTTS(text=text, lang='en', slow=slow, tld=tld)
            tts.save(filepath)
            print(f"gTTS: {emotion} -> TLD: {tld}, Slow: {slow}")
            return url_for('static', filename=f'audio/{filename}')
        except Exception as e:
            print(f"gTTS synthesis failed: {e}")
            return None
    
    def synthesize_pyttsx3(self, text, emotion, filename):
        """Offline TTS using pyttsx3"""
        if not PYTTSX3_AVAILABLE:
            return None
            
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            import pyttsx3
            engine = pyttsx3.init()
            
            voice_settings = self.get_voice_for_emotion(emotion, 'pyttsx3')
            engine.setProperty('rate', voice_settings['rate'])
            engine.setProperty('volume', voice_settings['volume'])
            
            # Try to set appropriate voice
            voices = engine.getProperty('voices')
            if voices and len(voices) > 1:
                if 'Positive' in emotion and len(voices) > 1:
                    engine.setProperty('voice', voices[1].id)
                elif 'Negative' in emotion:
                    engine.setProperty('voice', voices[0].id)
            
            engine.save_to_file(text, filepath)
            engine.runAndWait()
            engine.stop()
            
            print(f"pyttsx3: {emotion} -> rate: {voice_settings['rate']}")
            return url_for('static', filename=f'audio/{filename}')
            
        except Exception as e:
            print(f"pyttsx3 synthesis failed: {e}")
            return None
    
    def synthesize_google_cloud(self, text, emotion, filename):
        """Google Cloud Text-to-Speech"""
        if not GOOGLE_CLOUD_AVAILABLE or 'google_cloud' not in clients:
            return None
        
        voice_name, gender = self.get_voice_for_emotion(emotion, 'google_cloud')
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            synthesis_input = texttospeech.SynthesisInput(text=text)
            voice = texttospeech.VoiceSelectionParams(
                language_code="en-US",
                name=voice_name,
                ssml_gender=getattr(texttospeech.SsmlVoiceGender, gender)
            )
            
            speaking_rate = 1.2 if 'Very Positive' in emotion else 0.8 if 'Very Negative' in emotion else 1.0
            
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=speaking_rate
            )
            
            response = clients['google_cloud'].synthesize_speech(
                input=synthesis_input, voice=voice, audio_config=audio_config
            )
            
            with open(filepath, "wb") as out:
                out.write(response.audio_content)
            
            print(f"Google Cloud TTS: {emotion} -> {voice_name}")
            return url_for('static', filename=f'audio/{filename}')
        except Exception as e:
            print(f"Google Cloud TTS synthesis failed: {e}")
            return None
    
    def synthesize_macos_say(self, text, emotion, filename):
        """macOS built-in 'say' command"""
        if sys.platform != 'darwin':
            return None
            
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            voice_settings = self.get_voice_for_emotion(emotion, 'macos_say')
            voice = voice_settings['voice']
            rate = voice_settings['rate']
            
            # Change extension for macOS compatibility
            if filepath.endswith('.mp3'):
                filepath = filepath.replace('.mp3', '.m4a')
                filename = filename.replace('.mp3', '.m4a')
            
            cmd = [
                'say', '-v', voice, '-r', str(rate), '-o', filepath,
                '--data-format=mp4f', text
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                print(f"macOS say: {emotion} -> {voice} (rate: {rate})")
                return url_for('static', filename=f'audio/{filename}')
            else:
                print(f"macOS say failed: {result.stderr}")
                return None
                
        except Exception as e:
            print(f"macOS say synthesis failed: {e}")
            return None
    
    def synthesize(self, text, emotion, filename):
        """Main synthesis method with fallback logic"""
        if not filename.endswith(('.mp3', '.wav', '.m4a')):
            filename = filename.rsplit('.', 1)[0] + '.mp3'
        
        # Try primary TTS engine
        result = None
        if Config.TTS_ENGINE == 'gtts':
            result = self.synthesize_gtts(text, emotion, filename)
        elif Config.TTS_ENGINE == 'pyttsx3':
            result = self.synthesize_pyttsx3(text, emotion, filename)
        elif Config.TTS_ENGINE == 'google_cloud':
            result = self.synthesize_google_cloud(text, emotion, filename)
        elif Config.TTS_ENGINE == 'macos_say':
            result = self.synthesize_macos_say(text, emotion, filename)
        
        # Fallback logic
        if result is None:
            print(f"Primary TTS ({Config.TTS_ENGINE}) failed, trying fallbacks...")
            
            # Try configured fallback
            if Config.FALLBACK_TTS != Config.TTS_ENGINE:
                if Config.FALLBACK_TTS == 'gtts':
                    result = self.synthesize_gtts(text, emotion, filename)
                elif Config.FALLBACK_TTS == 'pyttsx3':
                    result = self.synthesize_pyttsx3(text, emotion, filename)
                elif Config.FALLBACK_TTS == 'macos_say':
                    result = self.synthesize_macos_say(text, emotion, filename)
            
            # Additional fallbacks
            if result is None:
                fallback_engines = []
                if GTTS_AVAILABLE and Config.TTS_ENGINE != 'gtts':
                    fallback_engines.append('gtts')
                if sys.platform == 'darwin' and Config.TTS_ENGINE != 'macos_say':
                    fallback_engines.append('macos_say')
                if PYTTSX3_AVAILABLE and Config.TTS_ENGINE != 'pyttsx3':
                    fallback_engines.append('pyttsx3')
                
                for engine in fallback_engines:
                    if engine == 'gtts':
                        result = self.synthesize_gtts(text, emotion, filename)
                    elif engine == 'macos_say':
                        result = self.synthesize_macos_say(text, emotion, filename)
                    elif engine == 'pyttsx3':
                        result = self.synthesize_pyttsx3(text, emotion, filename)
                    
                    if result is not None:
                        break
        
        return result

# Initialize engines
emotion_analyzer = EmotionAnalyzer()
tts_engine = TTSEngine()

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/synthesize', methods=['POST'])
def synthesize_route():
    data = request.get_json()
    text_input = data.get('text')
    
    if not text_input:
        return jsonify({'error': 'No text provided'}), 400
    
    # Limit text length
    if len(text_input) > Config.MAX_TEXT_LENGTH:
        text_input = text_input[:Config.MAX_TEXT_LENGTH]
    
    try:
        # Analyze emotion
        emotion_results = emotion_analyzer.analyze(text_input)
        primary_emotion = emotion_results['primary']
        
        # Generate unique filename
        unique_filename = f"output_{uuid.uuid4().hex}.mp3"
        
        # Synthesize speech
        audio_url = tts_engine.synthesize(text_input, primary_emotion['emotion'], unique_filename)
        
        if audio_url is None:
            return jsonify({'error': 'Failed to generate audio with any available TTS engine.'}), 500
        
        # Prepare response
        response_data = {
            'audio_url': audio_url,
            'emotion': primary_emotion['emotion'],
            'emotion_scores': {
                'compound': primary_emotion.get('scores', {}).get('compound', primary_emotion.get('confidence', 0)),
                'positive': primary_emotion.get('scores', {}).get('positive', 0),
                'negative': primary_emotion.get('scores', {}).get('negative', 0),
                'neutral': primary_emotion.get('scores', {}).get('neutral', 0),
                'intensity': primary_emotion.get('confidence', 0)
            },
            'analysis_details': {
                'primary_engine': primary_emotion['engine'],
                'tts_engine': Config.TTS_ENGINE,
                'all_analyses': emotion_results if Config.EMOTION_ENGINE == 'ensemble' else None
            }
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Synthesis route error: {e}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/config')
def get_config():
    """Configuration and status endpoint"""
    available_engines = {
        'emotion': {
            'vader': True,
            'huggingface': HF_AVAILABLE,
            'textblob': TEXTBLOB_AVAILABLE
        },
        'tts': {
            'gtts': GTTS_AVAILABLE,
            'pyttsx3': PYTTSX3_AVAILABLE,
            'google_cloud': GOOGLE_CLOUD_AVAILABLE and bool(Config.GOOGLE_CLOUD_KEY_PATH),
            'macos_say': sys.platform == 'darwin'
        }
    }
    
    return jsonify({
        'current_config': {
            'emotion_engine': Config.EMOTION_ENGINE,
            'tts_engine': Config.TTS_ENGINE,
            'fallback_tts': Config.FALLBACK_TTS,
            'max_text_length': Config.MAX_TEXT_LENGTH
        },
        'available_engines': available_engines,
        'system_info': {
            'platform': sys.platform,
            'python_version': sys.version
        }
    })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'tts_available': any([
            GTTS_AVAILABLE,
            PYTTSX3_AVAILABLE,
            sys.platform == 'darwin',
            GOOGLE_CLOUD_AVAILABLE and bool(Config.GOOGLE_CLOUD_KEY_PATH)
        ]),
        'emotion_available': True
    })

if __name__ == '__main__':
    print("=" * 50)
    print("🎤 EMPATHY ENGINE - EMOTION-AWARE TTS")
    print("=" * 50)
    print(f"Emotion Engine: {Config.EMOTION_ENGINE}")
    print(f"TTS Engine: {Config.TTS_ENGINE} (fallback: {Config.FALLBACK_TTS})")
    print(f"Platform: {sys.platform}")
    print(f"Max Text Length: {Config.MAX_TEXT_LENGTH} characters")
    print()
    print("Available Engines:")
    print(f"  - VADER Sentiment: ✅")
    print(f"  - Hugging Face: {'✅' if HF_AVAILABLE else '❌'}")
    print(f"  - TextBlob: {'✅' if TEXTBLOB_AVAILABLE else '❌'}")
    print(f"  - gTTS: {'✅' if GTTS_AVAILABLE else '❌'}")
    print(f"  - pyttsx3: {'✅' if PYTTSX3_AVAILABLE else '❌'}")
    print(f"  - macOS say: {'✅' if sys.platform == 'darwin' else '❌'}")
    print(f"  - Google Cloud: {'✅' if GOOGLE_CLOUD_AVAILABLE and Config.GOOGLE_CLOUD_KEY_PATH else '❌'}")
    print("=" * 50)
    
    # Create directories
    os.makedirs('static/audio', exist_ok=True)
    
    # Auto-find free port
    import socket
    
    def find_free_port(start_port=5000):
        for port in range(start_port, start_port + 10):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('localhost', port))
                    return port
            except OSError:
                continue
        return 8080
    
    port = find_free_port()
    print(f"🚀 Starting server on http://localhost:{port}")
    print("💡 Ready for emotion-aware text-to-speech!")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=port)