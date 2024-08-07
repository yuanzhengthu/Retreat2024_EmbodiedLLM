def generate_unique_filename(base_name):
    current_time = time.strftime("%Y%m%d-%H%M%S")
    return f"{base_name}_{current_time}.mp3"

import openai
from gtts import gTTS
from openai import OpenAI
import requests
import os
from pynput import keyboard
import cv2
import threading
from google.cloud import speech_v1p1beta1 as speech
import pyaudio
from six.moves import queue
import time
import pygame  # Import pygame for audio playback
import os
import time
import textwrap

# Function to play sound using pygame
def play_sound(filename):
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():  # Wait for audio to finish playing
        pygame.time.Clock().tick(10)


def clean_text_for_tts(text):
    return text.replace('#', '')


def upload_image_to_imgur(image_path, client_id):
    headers = {"Authorization": f"Client-ID {client_id}"}
    api_url = "https://api.imgur.com/3/upload"

    with open(image_path, "rb") as image_file:
        payload = {"image": image_file.read()}
        response = requests.post(api_url, headers=headers, files=payload)

    if response.status_code == 200:
        data = response.json()
        return data["data"]["link"]
    else:
        raise Exception(f"Failed to upload image: {response.status_code}, {response.text}")

def capture_image_from_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return None

    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        cap.release()
        return None

    image_path = "./inputs_camera/captured_image.jpg"
    cv2.imwrite(image_path, frame)
    cap.release()
    cv2.destroyAllWindows()
    return image_path

def upload_images_from_directory(directory_path, client_id):
    image_urls = {}
    image_paths = {}
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            image_path = os.path.join(directory_path, filename)
            try:
                image_url = upload_image_to_imgur(image_path, client_id)
                image_urls[filename] = image_url
                image_paths[filename] = image_path
                print(f"Image uploaded successfully: {image_url}")
            except Exception as e:
                print(f"Failed to upload {filename}: {str(e)}")
    return image_urls, image_paths

def format_conversation(history):
    system_message = history[0]['content']
    formatted_conversation = [f"{system_message}\n"]

    if len(history) > 1:
        summarized_history = summarize_conversation(history[1:-1])
        formatted_conversation.append(f"Summary of previous conversations:\n{summarized_history}\n")

    current_query = f"Now I want to know: {history[-1]['content']}"
    formatted_conversation.append(current_query)

    return "\n".join(formatted_conversation)

# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms

class MicrophoneStream(object):
    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk

        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
            stream_callback=self._fill_buffer,
        )

        self.closed = False
        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b"".join(data)

def listen_print_loop(responses, stop_event, width=70):
    # Accumulate all the transcript parts in a list
    transcript_parts = []

    for response in responses:
        if stop_event.is_set():
            break

        if not response.results:
            continue

        result = response.results[0]
        if not result.alternatives:
            continue

        transcript = result.alternatives[0].transcript
        transcript_parts.append(transcript)

        # Print the current combined transcript
        full_transcript = ' '.join(transcript_parts)
        # print("\rParts Transcript: {}".format(full_transcript), end='')
        wrapped_transcript = textwrap.fill(full_transcript, width=width)
        print("\rParts Transcript: {}".format(wrapped_transcript), end='')
    return ' '.join(transcript_parts)

# def on_press(key, stop_event, stream):
#     try:
#         if key.char == '1':
#             stop_event.set()
#             stream.closed = True  # Manually close the stream
#             return False
#     except AttributeError:
#         pass

def on_press(key, stop_event, start_event, stream):
    try:
        if key.char == '1':
            start_event.set()  # Start recording
            stop_event.clear()
            print("Recording started")
        elif key.char == '2':
            stop_event.set()  # Stop recording
            start_event.clear()
            if stream:
                stream.closed = True  # Manually close the stream
            print("Recording stopped")
    except AttributeError:
        pass


def periodic_input(stop_event, stream):
    while not stop_event.is_set():
        stream._buff.put(b'\x00' * CHUNK)
        time.sleep(1)

# def trasnscript_v2t():
#     client_google = speech.SpeechClient()
#
#     config = speech.RecognitionConfig(
#         encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
#         sample_rate_hertz=RATE,
#         language_code="en-US",
#     )
#
#     streaming_config = speech.StreamingRecognitionConfig(config=config)
#
#     # start_event = threading.Event()
#     stop_event = threading.Event()
#
#     # Create a separate thread to listen for key presses
#
#     listener = keyboard.Listener(on_press=lambda key: on_press(key, stop_event, stream))
#     listener.start()
#
#     with MicrophoneStream(RATE, CHUNK) as stream:
#         audio_generator = stream.generator()
#         requests = (speech.StreamingRecognizeRequest(audio_content=content) for content in audio_generator)
#
#         # Create a thread to periodically insert silent chunks
#         periodic_thread = threading.Thread(target=periodic_input, args=(stop_event, stream))
#         periodic_thread.start()
#
#         responses = client_google.streaming_recognize(streaming_config, requests)
#
#         full_transcript = listen_print_loop(responses, stop_event)
#     print("\nFull Transcript: {}".format(full_transcript))
#
#     stop_event.set()
#     listener.join()
#     periodic_thread.join()
#     return full_transcript

def trasnscript_v2t(start_event, stop_event, width=70):
    client_google = speech.SpeechClient()

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code="en-US",
    )

    streaming_config = speech.StreamingRecognitionConfig(config=config)

    while True:
        while not start_event.is_set():  # Wait until the start event is set
            time.sleep(0.1)

        with MicrophoneStream(RATE, CHUNK) as stream:
            listener = keyboard.Listener(on_press=lambda key: on_press(key, stop_event, start_event, stream))
            listener.start()

            audio_generator = stream.generator()
            requests = (speech.StreamingRecognizeRequest(audio_content=content) for content in audio_generator)

            stop_event.clear()  # Reset stop event before starting recording
            responses = client_google.streaming_recognize(streaming_config, requests)

            full_transcript = listen_print_loop(responses, stop_event)
            wrapped_transcript = textwrap.fill(full_transcript, width=width)
            # print("\nFull Transcript: {}".format(full_transcript))
            print("\rFull Transcript: {}".format(wrapped_transcript), end='')
            listener.stop()

        if not start_event.is_set() and stop_event.is_set():
            break

    return full_transcript



def summarize_conversation(history):
    summary_prompt = "Summarize the following conversation in a concise manner:\n\n"
    conversation_text = ""
    for message in history:
        if message['role'] == 'user':
            conversation_text += f"User: {message['content']}\n"
        elif message['role'] == 'assistant':
            conversation_text += f"Assistant: {message['content']}\n"
    summary_prompt += conversation_text

    messages = [
        {"role": "system", "content": "You are a summarization assistant."},
        {"role": "user", "content": summary_prompt}
    ]

    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.7,
    )
    summary = response.choices[0].message.content
    return summary


def chat_with_openai_camera(filename, url, conversation_history, width=70):
    user_message = trasnscript_v2t(start_event, stop_event)
    conversation_history.append({"role": "user", "content": user_message})
    formatted_conversation_history = format_conversation(conversation_history)
    if not user_message:

        return "Sorry, I could not understand your speech. Please try again."

    if user_message.lower() in ["exit", "stop", "quit"]:
        print("Exiting chat...")
        return "Exiting chat"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": formatted_conversation_history},
                {"type": "image_url", "image_url": {"url": url, "detail": "auto"}}
            ]
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=100,
    )
    response_openai = response.choices[0].message.content
    wrapped_transcript = textwrap.fill(response_openai, width=width)
    print(wrapped_transcript)

    return response_openai, user_message


def chat_with_openai_text(initial_message, summary_interval, language):
    conversation_history = [{"role": "system",
                             "content": "You are SenSay, a doctor. I will provide you with detailed information on the patient, and your task is to use the latest AI tools, such as medical imaging software and other machine learning programs, to diagnose the most likely cause of their symptoms. You should also incorporate traditional methods such as physical examinations, lab tests, and others into your assessment process to ensure accuracy. In this case, could you role-play as a doctor and have a conversation with me as a patient? Please provide concise responses."}]
    # You are a medical assistant. Please remember to answer more specifically and concisely! Don't answer point by point.
    if initial_message:
        conversation_history.append({"role": "user", "content": initial_message})

    while True:
        answer = ''
        if len(conversation_history) > 1:
            print("GPT:", conversation_history[-1]['content'])
        # transcript voice into text
        user_message = trasnscript_v2t(start_event, stop_event)

        if not user_message:
            continue

        if user_message.lower() in ["exit", "stop", "quit"]:
            print("Exiting chat...")
            break

        # Language switching logic
        if user_message.lower() in ["can you speak chinese?", "could you speak chinese?"]:
            language = 'zh'
            conversation_history.append({"role": "assistant", "content": "好的，我现在会用中文跟你交流。"})
            continue
        elif user_message.lower() in ["can you speak english?", "could you speak english?"]:
            language = 'en'
            conversation_history.append({"role": "assistant", "content": "Sure, I will now speak in English."})
            continue

        if any(phrase in user_message.lower() for phrase in
               ["open camera", "scan", "open eye", "open eyes", "open your camera", "open your eye", "open your eyes"]):
            # Switch to camera mode
            client_id = "93f5d15617cd538"
            image_path = capture_image_from_camera()
            tts = gTTS(text="Sure! Please input your name and allow me to access your health checkup report from today", lang=language)
            unique_filename = generate_unique_filename('response')
            tts.save(unique_filename)
            play_sound(unique_filename)
            if image_path:
                image_url = upload_image_to_imgur(image_path, client_id)
                answer, user_message = chat_with_openai_camera("captured_image.jpg", image_url, conversation_history)
                clean_answer = clean_text_for_tts(answer)
                # Using gTTS to vocalize the response
                tts = gTTS(text=clean_answer, lang=language)
                unique_filename = generate_unique_filename('response')
                tts.save(unique_filename)
                play_sound(unique_filename)
                conversation_history.append({"role": "assistant", "content": answer})
            continue

        try:
            conversation_history.append({"role": "user", "content": user_message})

            # Summarize conversation periodically
            if len(conversation_history) > summary_interval and len(conversation_history) % summary_interval == 0:
                summary = summarize_conversation(conversation_history)
                conversation_history.append(
                    {"role": "system", "content": f"Summary of previous conversations: {summary}"})

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=conversation_history,
                max_tokens=100
            )
            answer = response.choices[0].message.content
            conversation_history.append({"role": "assistant", "content": answer})
            clean_answer = clean_text_for_tts(answer)
            # Using gTTS to vocalize the response
            tts = gTTS(text=clean_answer, lang=language)
            unique_filename = generate_unique_filename('response')
            tts.save(unique_filename)
            play_sound(unique_filename)

        except Exception as e:
            print("Error during conversation:", e)
            break