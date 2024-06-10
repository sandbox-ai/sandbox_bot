

To transcribe and summarize:

```
python -m venv <envname>
source <envname>/bin/activate # Or on windows: .\<envname>\Scripts\activate
pip install -r requirements.txt
export GROQ_API_KEY="<your_groq_api_key>" # get at https://console.groq.com/keys 
python transcribe_and_summarize.py --audio-path <path-to-audio-file> --model <model-name> # Model name defaults to llama3-70b # Check available models at https://console.groq.com/docs/models
```






To use the general Assistant (UNFINISHED) : install LLM studio (or Ollama), and then do the following: 
```
    git clone https://github.com/myshell-ai/OpenVoice
    cd OpenVoice
    git clone https://huggingface.co/myshell-ai/OpenVoice
    cp -r OpenVoice/* .
    rm -rf OpenVoice
    cd ..
    rm OpenVoice/README.md
    cp -r OpenVoice/* .
    rm -rf OpenVoice
    sudo apt install ffmpeg
    pip install -r requirements.txt
    pip install whisper pynput pyaudio
    python sandbox_bot.py
```
