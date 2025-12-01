"""
Text-to-Speech utility functions for robot voice feedback
"""

import pyttsx3
import colorlog

# Setup logger
logger = colorlog.getLogger(__name__)

# Initialize the TTS engine globally (lazy initialization)
_tts_engine = None


def get_tts_engine():
    """
    Get or initialize the TTS engine (singleton pattern).
    
    Returns:
        pyttsx3.Engine: The TTS engine instance
    """

    # --- Use the ID you found! ---
    GREEK_VOICE_ID = "grk/el" 
    # -----------------------------

    global _tts_engine
    if _tts_engine is None:
        try:
            _tts_engine = pyttsx3.init()
            # Optional: Configure voice properties
            _tts_engine.setProperty('rate', 150)  # Speed of speech (default ~200)
            _tts_engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)
            # Set the Greek voice ID
            # _tts_engine.setProperty('voice', GREEK_VOICE_ID)

            logger.info("TTS engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize TTS engine: {e}")
            _tts_engine = None
    return _tts_engine


def speak(text, blocking=True):
    """
    Speak the given text using text-to-speech.
    
    Args:
        text: String to speak
        blocking: If True, waits for speech to complete before returning
    
    Returns:
        bool: True if speech was successful, False otherwise
    """
    engine = get_tts_engine()
    if engine is None:
        logger.warning(f"TTS unavailable, would have said: {text}")
        return False
    
    try:
        engine.say(text)
        if blocking:
            engine.runAndWait()
        return True
    except Exception as e:
        logger.error(f"TTS error: {e}")
        return False


def announce_target_found(message):
    """
    Announce when a target object is found.
    
    Args:
        target_class: Name of the detected object (e.g., 'chair', 'ball', 'person')
    
    Returns:
        bool: True if announcement was successful
    """
    # message = "Announce when a target object is found.Announce when a target object is found.Announce when a target object is found.Announce when a target object is found.Announce when a target object is found. "
    logger.info(f"🔊 Speaking: {message}")
    return speak(message, blocking=True)  # Non-blocking so robot can continue


def announce_target_reached(target_class):
    """
    Announce when the robot has reached the target.
    
    Args:
        target_class: Name of the target object
    
    Returns:
        bool: True if announcement was successful
    """
    message = f"Reached {target_class}"
    logger.info(f"🔊 Speaking: {message}")
    return speak(message, blocking=True)  # Blocking for final announcement


def announce_searching(target_class):
    """
    Announce that the robot is searching for a target.
    
    Args:
        target_class: Name of the target object to search for
    
    Returns:
        bool: True if announcement was successful
    """
    message = f"Searching for {target_class}"
    logger.info(f"🔊 Speaking: {message}")
    return speak(message, blocking=False)


def announce_obstacle(obstacle_class):
    """
    Announce obstacle detection.
    
    Args:
        obstacle_class: Name of the detected obstacle
    
    Returns:
        bool: True if announcement was successful
    """
    message = f"Avoiding {obstacle_class}"
    logger.info(f"🔊 Speaking: {message}")
    return speak(message, blocking=False)
