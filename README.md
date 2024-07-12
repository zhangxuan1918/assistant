# Assistant

<video width="600" controls>
  <source src="docs/demo_480p.mov" type="video/quicktime">
  Demo.
</video>

* STT server: https://github.com/fedirz/faster-whisper-server
* TTS server: https://github.com/timhagel/MeloTTS-Docker-API-Server	
* LLM server: https://github.com/ollama/ollama

## How To Run

1. start servers (GPU, 10G mem is sufficient, I use RTX3090)
   * start ollama: `ollama run llama3`
   * start faster whisper server: 
      * `curl -sO https://raw.githubusercontent.com/fedirz/faster-whisper-server/master/compose.yaml`
      * `docker compose up --detach faster-whisper-server-cuda`
   * start melo tts server:
      * `docker pull timhagel/melotts-api-server`
      * `docker run --name melotts-server -p 8888:8080 --gpus=all -e DEFAULT_SPEED=1 -e DEFAULT_LANGUAGE=EN -e DEFAULT_SPEAKER_ID=EN-Default timhagel/melotts-api-server`
2. run assistant: `python run.py`
3. copy some text, e.g. web page, as context to clipboard: `ctrl c`
4. press key `ESC`, ask your question and press key `ESC` to stop recording
5. wait for the answer in voice