import pyttsx3
import time
import logging
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpeechEngine:
    
    def __init__(self, cooldown=3.0):
        self.enabled = True
        self.engine = None
        self.speech_thread = None
        self.is_speaking = False
        
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty("rate", 175)
            self.engine.setProperty("volume", 1.0)
            logger.info("[Speech] Engine initialized successfully")
        except Exception as e:
            logger.warning(f"[Speech] Failed to initialize TTS: {e}")
            logger.warning("[Speech] Continuing without audio")
            self.enabled = False

        self.cooldown = cooldown
        self.last_spoken_time = 0.0
        self.last_message = None
        self.last_direction = None
        self.last_distance = None
        
        self.distance_priority = {
            "very_close": 3,
            "close": 2,
            "far": 1
        }
        
        if self.enabled:
            logger.info("[Speech] Ready for Obstacle Avoidance")
        else:
            logger.info("[Speech] Running in silent mode (no audio device)")

    def stop(self):
        if not self.enabled:
            return
            
        if self.speech_thread and self.speech_thread.is_alive():
            self.speech_thread.join(timeout=1.0)
        
        try:
            if self.engine:
                self.engine.stop()
            logger.info("[Speech] Engine stopped")
        except Exception as e:
            logger.debug(f"[Speech] Stop error: {e}")

    def _speak_thread(self, message):
        try:
            self.is_speaking = True
            self.engine.say(message)
            self.engine.runAndWait()
        except Exception as e:
            logger.warning(f"[Speech] Error during speech: {e}")
        finally:
            self.is_speaking = False

    def speak(self, direction, distance, obstacle_class=None):
        if not self.enabled:
            return  # Silent mode
            
        message = self._build_message(direction, distance, obstacle_class)
        if not message:
            return

        now = time.time()
        elapsed = now - self.last_spoken_time

        min_cooldown = 0.5 if distance == "very_close" else 2.5
        
        got_closer = False
        if self.last_distance:
            got_closer = self.distance_priority.get(distance, 0) > self.distance_priority.get(self.last_distance, 0)

        if self.is_speaking:
            return

        if elapsed >= min_cooldown or got_closer:
            # Start speech in background thread
            self.speech_thread = threading.Thread(
                target=self._speak_thread,
                args=(message,),
                daemon=True
            )
            self.speech_thread.start()
            
            self.last_spoken_time = now
            self.last_distance = distance
            self.last_direction = direction

    def _build_message(self, direction, distance, obstacle_class=None):
        prefix = f"{obstacle_class} " if obstacle_class else ""
        if distance == "very_close":
            return f"Danger! {prefix}{direction} very close!"
        if distance == "close":
            return f"{prefix}Ahead {direction}."
        return ""