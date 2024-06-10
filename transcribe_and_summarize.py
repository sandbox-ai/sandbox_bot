import os
import json
import sys
import argparse
from groq import Groq

# Set up parser
parser = argparse.ArgumentParser(description="Transcribe and summarize audio meetings.")
parser.add_argument('--audio-path', type=str, required=True, help='Path to the audio file')
parser.add_argument('--model', type=str, required=False, default="llama3-70b-8192", help="Language model to use")


# Parse arguments
args = parser.parse_args()
audio_path = args.audio_path
groq_model = args.model

# Transcribe audio
transcribe_command = "insanely-fast-whisper --file-name " + audio_path

try:
    os.system(transcribe_command)
except: 
    print("audio transcription failed")


with open('output.json', mode='r') as text:
    transcription = json.load(text)

text = transcription['text']


# Run LLM
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

PROMPT_init = "Sos un experto en hacer resúmenes de reuniones de trabajo de una empresa de Argentina, cuyos integrantes son Dani, Facundo, Francis, Marian, Rafael y Pablo. Tu tarea es leer el transcripto de una reunión de esta empresa y hacer un RESUMEN DETALLADO de la reunión, teniendo en cuenta lo que dijo cada uno de los integrantes. También tenés que hacer una lista de bulletpoint con los principales takeaways de la reunión. \n ------------- \n Transcripción de la reunión: \n "

PROMPT_end = "A continuación, RESUMEN y principales TAKEAWAYS de la reunión: \n ------------------- \n "

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": PROMPT_init + text + PROMPT_end
        }
    ],
    model=groq_model,
)

output = chat_completion.choices[0].message.content
print(output)
print("----------------- \n Resumen y takeaways en: output.txt")


with open('output.txt', 'w') as file:
    file.write(output)









