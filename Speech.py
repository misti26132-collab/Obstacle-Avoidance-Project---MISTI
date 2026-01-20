import pyttsx3
import time


class SpeechEngine:
    def __init__(self, cooldown=3.0):
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 175)  # Slightly faster for urgency
        self.engine.setProperty("volume", 1.0)

        self.cooldown = cooldown
        self.last_spoken_time = 0.0
        self.last_message = None
        self.last_direction = None
        self.last_distance = None
        self.distance_priority = {"very_close": 3, "close": 2, "far": 1}

        print("[Speech] Engine initialized (main-thread mode)")

    def speak(self, direction, distance):
        message = self._build_message(direction, distance)
        if not message:
            return

        now = time.time()
        elapsed = now - self.last_spoken_time

        # Dynamic cooldown based on urgency
        if distance == "very_close":
            min_cooldown = 0.5  # Repeat urgent warnings quickly
        elif distance == "close":
            min_cooldown = 2.0  # Moderate frequency
        else:
            min_cooldown = 4.0  # Infrequent for far objects

        # Check if object got closer (escalation)
        got_closer = False
        if self.last_distance is not None and distance is not None:
            current_priority = self.distance_priority.get(distance, 0)
            last_priority = self.distance_priority.get(self.last_distance, 0)
            got_closer = current_priority > last_priority

        # Allow re-announcement if direction changed OR distance changed
        situation_changed = (direction != self.last_direction) or (distance != self.last_distance)
        
        # Speak if: enough time passed OR object got closer OR situation became critical
        should_speak = (
            elapsed >= min_cooldown or 
            got_closer or 
            (situation_changed and distance == "very_close")
        )
        
        if should_speak:
            try:
                print(f"[Speech] Speaking: {message} (got_closer={got_closer})")
                self.engine.say(message)
                self.engine.runAndWait()
                print("[Speech] Finished speaking")

                self.last_spoken_time = now
                self.last_message = message
                self.last_direction = direction
                self.last_distance = distance

            except Exception as e:
                print(f"[Speech] Speech error: {e}")

    def stop(self):
        try:
            self.engine.stop()
        except Exception:
            pass
        print("[Speech] Stopped")

    def _build_message(self, direction, distance):
        # Priority messages for very close obstacles
        if distance == "very_close":
            if direction == "center":
                return "Stop! Obstacle very close ahead!"
            elif direction == "left":
                return "Danger! Very close on the left!"
            elif direction == "right":
                return "Danger! Very close on the right!"
        
        # Close obstacles
        if distance == "close":
            if direction == "center":
                return "Obstacle ahead."
            elif direction == "left":
                return "Obstacle on the left."
            elif direction == "right":
                return "Obstacle on the right."
        
        # Far obstacles - minimal warnings
        if distance == "far":
            if direction == "center":
                return "Something ahead."
            # Don't warn about far left/right obstacles
            return ""
        
        return ""