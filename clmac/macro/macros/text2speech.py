import pyperclip
import pyttsx3

engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)
engine.setProperty('rate', 250)

def text2speech() -> None:
    text = pyperclip.paste()
    engine.say(text)
    engine.runAndWait()
