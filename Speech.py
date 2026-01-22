import pyttsx3
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpeechEngine:
    def __init__(self, cooldown=3.0):
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty("rate", 175)  # Slightly faster for urgency
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
        
        # Priority levels for distance urgency
        self.distance_priority = {
            "very_close": 3,
            "close": 2,
            "far": 1
        }

        logger.info("[Speech] Using main-thread mode for reliability")

    def speak(self, direction, distance, obstacle_class=None):

        message = self._build_message(direction, distance, obstacle_class)
        if not message:
            return

        now = time.time()
        elapsed = now - self.last_spoken_time

        # Dynamic cooldown based on urgency
        if distance == "very_close":
            min_cooldown = 0.5  
        elif distance == "close":
            min_cooldown = 2.0  
        else:
            min_cooldown = 4.0  

        got_closer = False
        if self.last_distance is not None and distance is not None:
            current_priority = self.distance_priority.get(distance, 0)
            last_priority = self.distance_priority.get(self.last_distance, 0)
            got_closer = current_priority > last_priority

        # Allow re-announcement if direction changed OR distance changed
        situation_changed = (
            direction != self.last_direction or 
            distance != self.last_distance
        )
        
        # Speak if: enough time passed OR object got closer OR situation became critical
        should_speak = (
            elapsed >= min_cooldown or 
            got_closer or 
            (situation_changed and distance == "very_close")
        )
        
        if should_speak:
            try:
                logger.info(f"[Speech] Speaking: '{message}' (escalation={got_closer})")
                
                # FIXED: Stop any ongoing speech first to prevent hanging
                try:
                    self.engine.stop()
                except:
                    pass
                
                self.engine.say(message)
                self.engine.runAndWait()
                logger.debug("[Speech] Finished speaking")

                # Update state
                self.last_spoken_time = now
                self.last_message = message
                self.last_direction = direction
                self.last_distance = distance

            except RuntimeError as e:
                # FIXED: Handle specific "run loop already started" error
                if "run loop already started" in str(e):
                    logger.warning("[Speech] Engine busy, skipping this message")
                else:
                    logger.error(f"[Speech] Speech error: {e}")
            except Exception as e:
                logger.error(f"[Speech] Speech error: {e}")

    def stop(self):
        """
        Stop the speech engine and clean up
        """
        try:
            self.engine.stop()
            logger.info("[Speech] Engine stopped")
        except Exception as e:
            logger.warning(f"[Speech] Error stopping engine: {e}")

    def _build_message(self, direction, distance, obstacle_class=None):
        """
        Build appropriate warning message based on obstacle parameters
        
        Args:
            direction: Obstacle direction
            distance: Obstacle distance
            obstacle_class: Type of obstacle
            
        Returns:
            str: Warning message to speak
        """
        # Optional: Include obstacle class in message
        obstacle_prefix = f"{obstacle_class} " if obstacle_class else ""
        
        # Priority messages for very close obstacles
        if distance == "very_close":
            if direction == "center":
                return f"Stop! {obstacle_prefix}Obstacle very close ahead!"
            elif direction == "left":
                return f"Danger! {obstacle_prefix}Very close on the left!"
            elif direction == "right":
                return f"Danger! {obstacle_prefix}Very close on the right!"
        
        # Close obstacles
        if distance == "close":
            if direction == "center":
                return f"{obstacle_prefix}Obstacle ahead."
            elif direction == "left":
                return f"{obstacle_prefix}Obstacle on the left."
            elif direction == "right":
                return f"{obstacle_prefix}Obstacle on the right."
        
        # Far obstacles - minimal warnings
        if distance == "far":
            if direction == "center":
                return "Something ahead."
            # Don't warn about far left/right obstacles
            return ""
        
        return ""


def test_speech_engine():
    print("=" * 60)
    print("SPEECH ENGINE TEST")
    print("=" * 60)
    
    try:
        speaker = SpeechEngine(cooldown=2.0)
    except Exception as e:
        print(f"[ERROR] {e}")
        return
    
    test_cases = [
        ("center", "very_close", "person"),
        ("left", "very_close", "chair"),
        ("right", "very_close", "car"),
        ("center", "close", "person"),
        ("left", "close", "bench"),
        ("right", "close", "bicycle"),
        ("center", "far", "person"),
    ]
    
    print("\nTesting all scenarios (press Ctrl+C to stop):\n")
    
    try:
        for i, (direction, distance, obstacle) in enumerate(test_cases):
            print(f"[Test {i+1}/{len(test_cases)}] {direction} - {distance} - {obstacle}")
            speaker.speak(direction, distance, obstacle)
            time.sleep(3)  # Wait between tests
    
    except KeyboardInterrupt:
        print("\n\n[Interrupted]")
    
    finally:
        speaker.stop()
        print("[Done]")
        print("=" * 60)


if __name__ == "__main__":
    test_speech_engine()