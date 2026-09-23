import re

# Common STT mistranscriptions and their corrections
CORRECTIONS = {
    # If it hears the broken version, swap it for the real question
    r"ስምሽ\s+ለናግኝ\s+ትቻያለች": "ስምሽን ልትነግሪኝ ትችያለሽ?",
    r"ሰመችል\s+ግሪን\s+ትሻያለች": "ስምሽን ልትነግሪኝ ትችያለሽ?"
}

def correct_stt_transcript(transcript: str) -> str:
    corrected = transcript
    for wrong, right in CORRECTIONS.items():
        corrected = re.sub(wrong, right, corrected)
    return corrected
