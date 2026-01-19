import pyttsx3
import time

class SpeechEngine:
    def __init__(self, cooldown=2.5):
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 165)
        self.last_message = ""
        self.last_time = 0
        self.cooldown = cooldown

    def speak(self, direction, distance):
        message = self._build_message(direction, distance)
        now = time.time()

        if message and message != self.last_message and now - self.last_time > self.cooldown:
            self.engine.say(message)
            self.engine.runAndWait()
            self.last_message = message
            self.last_time = now

    def _build_message(self, direction, distance):
        if direction == "center":
            if distance == "very close":
                return "Stop. Obstacle very close ahead."
            return "Obstacle ahead."

        if direction == "left":
            return "Obstacle on the left."

        if direction == "right":
            return "Obstacle on the right."

        return ""
