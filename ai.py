import flet as ft
import cv2
import mediapipe as mp
import speech_recognition as sr
import threading
import g4f

# Function to query GPT4Free
def chatgpt_query(prompt):
    response = g4f.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response

class AIAssistant:
    def __init__(self, page):
        self.page = page
        self.recognizer = sr.Recognizer()
        self.camera = cv2.VideoCapture(0)
        self.hands = mp.solutions.hands.Hands()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.running = True

        # UI Elements
        self.page.title = "AI Assistant"
        self.page.window_width = 500
        self.page.window_height = 700

        self.voice_text = ft.Text("Listening...")
        self.status_text = ft.Text("Status: Idle")
        self.start_btn = ft.ElevatedButton("Start Voice Command", on_click=self.start_voice)
        self.stop_btn = ft.ElevatedButton("Stop", on_click=self.stop)

        self.page.add(self.voice_text, self.status_text, self.start_btn, self.stop_btn)

    def start_voice(self, e):
        threading.Thread(target=self.listen_voice).start()

    def listen_voice(self):
        with sr.Microphone() as source:
            self.status_text.value = "Listening..."
            self.page.update()
            try:
                audio = self.recognizer.listen(source)
                text = self.recognizer.recognize_google(audio)
                response = chatgpt_query(text)
                self.voice_text.value = f"ChatGPT: {response}"
            except sr.UnknownValueError:
                self.voice_text.value = "Could not understand"
            self.page.update()

    def start_camera(self):
        while self.running:
            ret, frame = self.camera.read()
            if not ret:
                break
            self.detect_hand(frame)
            self.detect_eye(frame)
        self.camera.release()
        cv2.destroyAllWindows()

    def detect_hand(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        if results.multi_hand_landmarks:
            self.status_text.value = "Hand Detected"
        self.page.update()

    def detect_eye(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            eyes = self.eyes_cascade.detectMultiScale(roi_gray)
            if len(eyes) > 0:
                self.status_text.value = "Eyes Detected"
                self.page.update()

    def stop(self, e):
        self.running = False
        self.camera.release()
        self.page.window_close()

def main(page: ft.Page):
    assistant = AIAssistant(page)
    threading.Thread(target=assistant.start_camera).start()

ft.app(target=main)
        
