import pyttsx3
import time


class SpeechEngine:
    def __init__(self, cooldown=1.0):
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 165)
        self.engine.setProperty("volume", 1.0)

        self.cooldown = cooldown
        self.last_spoken_time = 0.0
        self.last_message = None

        print("[Speech] Engine initialized (main-thread mode)")

    def speak(self, direction, distance):
        message = self._build_message(direction, distance)
        if not message:
            return

        # Prevent repeating the same message
        if message == self.last_message:
            return

        now = time.time()
        elapsed = now - self.last_spoken_time

        is_priority = "very close" in message
        min_cooldown = 0.3 if is_priority else self.cooldown

        if elapsed < min_cooldown:
            return

        try:
            print(f"[Speech] Speaking: {message}")
            self.engine.say(message)
            self.engine.runAndWait()
            print("[Speech] Finished speaking")

            self.last_spoken_time = now
            self.last_message = message

        except Exception as e:
            print(f"[Speech] Speech error: {e}")

    def stop(self):
        try:
            self.engine.stop()
        except Exception:
            pass
        print("[Speech] Stopped")

    def _build_message(self, direction, distance):
        if direction == "center":
            if distance == "very close":
                return "Stop. Obstacle very close ahead."
            return "Obstacle ahead."

        if direction == "left":
            if distance == "very close":
                return "Obstacle very close on the left."
            return "Obstacle on the left."

        if direction == "right":
            if distance == "very close":
                return "Obstacle very close on the right."
            return "Obstacle on the right."

        return ""
