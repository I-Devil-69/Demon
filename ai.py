import flet as ft
import os
import torch
import whisper
import openai
import cv2
import numpy as np
from deepface import DeepFace

# Initialize AI models
whisper_model = whisper.load_model("small")
openai.api_key = "YOUR_OPENAI_API_KEY"

# GUI Application
def main(page: ft.Page):
    page.title = "All-in-One AI System"
    page.window_width = 800
    page.window_height = 600

    # Voice Command Input
    def recognize_voice(e):
        with open("voice_input.wav", "rb") as audio:
            result = whisper_model.transcribe(audio)
            text_output.value = result["text"]
            page.update()

    voice_button = ft.ElevatedButton("Record Voice", on_click=recognize_voice)
    text_output = ft.Text("Recognized Speech: ")

    # Face Modification
    def modify_face(e):
        img = cv2.imread("input.jpg")
        modified_img = DeepFace.analyze(img, actions=["age", "gender"])
        cv2.imwrite("output.jpg", modified_img)
        face_output.src = "output.jpg"
        page.update()

    face_button = ft.ElevatedButton("Modify Face", on_click=modify_face)
    face_output = ft.Image(src="")

    # AI Chatbot
    def chatbot_response(e):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": chat_input.value}]
        )
        chat_output.value = response["choices"][0]["message"]["content"]
        page.update()

    chat_input = ft.TextField(label="Ask AI", width=300)
    chat_button = ft.ElevatedButton("Chat", on_click=chatbot_response)
    chat_output = ft.Text("Response: ")

    # Adding Components to UI
    page.add(
        ft.Column([
            voice_button, text_output,
            face_button, face_output,
            chat_input, chat_button, chat_output
        ])
    )

# Run the app
ft.app(target=main)
