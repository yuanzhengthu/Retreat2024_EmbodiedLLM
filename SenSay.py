import openai
from openai import OpenAI
from pynput import keyboard
import threading
import pygame  # Import pygame for audio playback
import os
from utils.utils import on_press, chat_with_openai_text
# Initialize Pygame
pygame.init()
pygame.mixer.init()

# Your OpenAI API key
openai.api_key = 'Your own api-key'
client = OpenAI(api_key=openai.api_key)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "GOOGLE_APPLICATION_CREDENTIALS, json file"

# Main loop to start detecting and responding
initial_message = "Start Conversation!"
summary_interval  = 5
language = 'en'  # Default language is English

# Using gTTS to vocalize the response
# tts = gTTS(text="Hi, Yuan, I am SenSay. What brings you in today?", lang=language)
# unique_filename = generate_unique_filename('response')
# tts.save(unique_filename)
# play_sound(unique_filename)
########################################################################################################
start_event = threading.Event()
stop_event = threading.Event()

# Create a separate thread to listen for key presses
listener = keyboard.Listener(on_press=lambda key: on_press(key, stop_event, start_event, None))
listener.start()
########################################################################################################
chat_with_openai_text(initial_message, summary_interval, language)

