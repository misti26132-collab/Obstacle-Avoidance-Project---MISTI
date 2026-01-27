import pyttsx3
import time
import logging
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpeechEngine:
    def __init__(self, cooldown=3.0):
        try:
            self.engine = pyttsx3.init() 
            self.engine.setProperty("rate", 175)
            self.engine.setProperty("volume", 1.0)
            logger.info("[Speech] Engine initialized successfully")
        except Exception as e:
            logger.error(f"[Speech] Failed to initialize TTS engine: {e}")
            raise

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
        logger.info("[Speech] Ready for Obstacle Avoidance")

    def stop(self):
        """Safely shuts down the engine"""
        try:
            self.engine.stop()
            logger.info("[Speech] Engine stopped")
        except Exception as e:
            logger.debug(f"[Speech] Stop called on inactive engine: {e}")

    def speak(self, direction, distance, obstacle_class=None, priority='HIGH'):
        message = self._build_message(direction, distance, obstacle_class, priority)
        if not message:
            return

        now = time.time()
        elapsed = now - self.last_spoken_time

        min_cooldown = 0.5 if distance == "very_close" else 2.5
        
        got_closer = False
        if self.last_distance:
            got_closer = self.distance_priority.get(distance, 0) > self.distance_priority.get(self.last_distance, 0)

        if elapsed >= min_cooldown or got_closer:
            try:
                self.engine.say(message)
                self.engine.runAndWait()
                self.last_spoken_time = now
                self.last_distance = distance
                self.last_direction = direction
            except RuntimeError:
                logger.warning("[Speech] Engine loop busy - skipping frame")

    def _build_message(self, direction, distance, obstacle_class=None, priority='HIGH'):
        try:
            friendly_name = config.FRIENDLY_NAMES.get(obstacle_class, obstacle_class)
        except ImportError:
            friendly_name = obstacle_class
        
        if not friendly_name:
            friendly_name = "obstacle"
        
        # Direction phrases
        direction_phrases = {
            'left': 'on your left',
            'center': 'directly ahead',
            'right': 'on your right'
        }
        direction_phrase = direction_phrases.get(direction, direction)
        
        if distance == "very_close":
            if priority == 'CRITICAL':
                return f"Stop! {friendly_name.capitalize()} {direction_phrase}!"
            else:
                return f"Warning! {friendly_name.capitalize()} {direction_phrase}!"
        
        elif distance == "close":
            if priority == 'CRITICAL':
                return f"{friendly_name.capitalize()} approaching {direction_phrase}"
            else:
                return f"{friendly_name.capitalize()} {direction_phrase}"
        
        else:
            return ""