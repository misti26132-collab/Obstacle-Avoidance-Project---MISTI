import pyttsx3
import time

class SpeechEngine:
    def __init__(self, cooldown=2.5):
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 165)
        self.engine.setProperty("volume", 1.0)
        self.last_spoken_message = None
        self.last_spoken_time = 0
        self.cooldown = cooldown

    def speak(self, direction, distance):
        message = self._build_message(direction, distance)
        
        if not message:
            return
        
        current_time = time.time()
        time_since_last = current_time - self.last_spoken_time
        
        should_speak = (
            message != self.last_spoken_message or 
            time_since_last >= self.cooldown
        )
        
        if should_speak:
            try:
                self.engine.say(message)
                self.engine.runAndWait()
                self.last_spoken_message = message
                self.last_spoken_time = current_time
            except Exception as e:
                print(f"Error speaking: {e}")

    def _build_message(self, direction, distance):
        """Build warning message based on direction and distance"""
        if direction == "center":
            if distance == "very close":
                return "Stop. Obstacle very close ahead."
            elif distance == "close":
                return "Warning. Obstacle ahead."
            else:
                return "Obstacle ahead."

        if direction == "left":
            if distance == "very close":
                return "Obstacle very close on the left."
            else:
                return "Obstacle on the left."

        if direction == "right":
            if distance == "very close":
                return "Obstacle very close on the right."
            else:
                return "Obstacle on the right."

        return ""