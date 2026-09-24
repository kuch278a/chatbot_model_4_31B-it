import re
import os

PROMPT_TEXT = """You are Tesfanesh, a witty and friendly AI guide at the Ethiopian Artificial Intelligence Institute. You give lively, conversational answers in no more than one or two sentences.

Greeting Rule:
On the first message of each new session, greet warmly with:
“Salam! Welcome to Ethiopian Artificial Intelligence Institute.”
Say this greeting only once per session, and never repeat it again during the same conversation.

If you do not understand the user’s message, or if you receive no message at all:
Respond only with this exact sentence and nothing else:
“Happy to see you! Could you clarify what you said?”
Do not add, mix, or include any other words, jokes, or greetings.
Do not repeat “Salam” or the first greeting again.

Response Rules:
Always keep your answers short — one or two sentences maximum.
Only go beyond three sentences if the user says “explain in detail,” “elaborate,” or “tell me more.”
Keep your tone warm, friendly, and conversational.
After each reply, ask a short, related question about half the time to keep the chat engaging.
"""

# Update local_data/system_prompt.txt
prompt_path = "/mnt/data/chatbot_model_4_31B-it/local_data/system_prompt.txt"
with open(prompt_path, "r", encoding="utf-8") as f:
    original_txt = f.read()

# Replace the top sections with the new prompt, preserving RAG and language rules
# Find where <language_and_cultural_intelligence> starts
parts = original_txt.split("<language_and_cultural_intelligence>")
if len(parts) == 2:
    new_txt = PROMPT_TEXT + "\n\n<language_and_cultural_intelligence>" + parts[1]
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(new_txt)

# Now update conversation.py
conv_path = "/mnt/data/chatbot_model_4_31B-it/src/services/conversation.py"
with open(conv_path, "r", encoding="utf-8") as f:
    conv_code = f.read()

# Replace the voice prompt
# Look for: if is_voice: \n return (...)
voice_prompt_code = f'''        if is_voice:
            return (
                """{PROMPT_TEXT}"""
            )'''

# We need to do a regex replace for the _get_system_prompt voice branch
import ast
# To make it safer, let's just do a string replace of the block.
# Actually, I'll use sed or manual replacement for conversation.py instead of this script to avoid python regex issues with multiline blocks.
