To use: install LLM studio (or Ollama), and then do the following: 
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
