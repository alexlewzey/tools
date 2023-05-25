from multiprocessing import Process

import pyperclip
import pyttsx3

engine = pyttsx3.init()
voices = engine.getProperty("voices")
engine.setProperty("voice", voices[1].id)
engine.setProperty("rate", 250)


def speak_clipboard():
    text = pyperclip.paste()
    engine.say(text)
    engine.runAndWait()


def text2speech() -> None:
    process = Process(target=speak_clipboard)
    process.start()
