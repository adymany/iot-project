def validate_voice_model_path(path):
    """Check if the voice model file exists at the given path."""
    return os.path.exists(path)

def validate_text_input(text):
    """Check if the provided text input is valid (non-empty)."""
    return bool(text.strip())

def get_absolute_path(relative_path):
    """Return the absolute path for a given relative path."""
    return os.path.abspath(relative_path)