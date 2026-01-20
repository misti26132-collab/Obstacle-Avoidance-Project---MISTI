import pyttsx3
import time
import threading
import queue
import sys

class SpeechEngine:
    def __init__(self, cooldown=1.5):
        # Create engine (will be re-initialized in thread)
        self.engine = None
        self.last_spoken_time = 0
        self.cooldown = cooldown
        self.speech_queue = queue.Queue()
        self.speech_thread = None
        self.should_stop = False
        
        # Start background speech thread
        self.speech_thread = threading.Thread(
            target=self._speech_worker,
            daemon=True
        )
        self.speech_thread.start()
        print("[Speech] Engine initialized")

    def _speech_worker(self):
        """Background worker thread for non-blocking speech"""
        # Create engine in this thread (thread-safe)
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty("rate", 165)
            self.engine.setProperty("volume", 1.0)
            print("[Speech] Worker thread started successfully")
        except Exception as e:
            print(f"[Speech] Failed to initialize speech engine: {e}")
            return
        
        while not self.should_stop:
            try:
                # Wait for messages with timeout
                message = self.speech_queue.get(timeout=1.0)
                if message is None:  # Sentinel to stop
                    break
                
                print(f"[Speech] Speaking: {message}")
                try:
                    self.engine.say(message)
                    self.engine.runAndWait()
                    print(f"[Speech] Finished speaking")
                except Exception as e:
                    print(f"[Speech] Error speaking: {e}")
                    # Try to reinitialize engine if it fails
                    try:
                        self.engine = pyttsx3.init()
                        self.engine.setProperty("rate", 165)
                        self.engine.setProperty("volume", 1.0)
                    except:
                        pass
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Speech] Worker error: {e}")

    def speak(self, direction, distance):
        message = self._build_message(direction, distance)
        
        if not message:
            return
        
        current_time = time.time()
        time_since_last = current_time - self.last_spoken_time
        
        # Always speak if enough time has passed since last speech
        # Priority messages (very close) have shorter cooldown
        is_priority = "very close" in message
        min_cooldown = 0.3 if is_priority else 0.8
        
        if time_since_last >= min_cooldown:
            print(f"[Speech] Queuing: {message} (time_since_last: {time_since_last:.2f}s)")
            self.speech_queue.put(message)
            self.last_spoken_time = current_time
        else:
            print(f"[Speech] Skipped (cooldown): {message} (waiting: {min_cooldown - time_since_last:.2f}s)")

    def stop(self):
        """Gracefully stop the speech engine"""
        print("[Speech] Stopping...")
        self.should_stop = True
        self.speech_queue.put(None)
        if self.speech_thread:
            self.speech_thread.join(timeout=2)
        print("[Speech] Stopped")

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

