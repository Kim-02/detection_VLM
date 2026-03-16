
source .venv/bin/activate
pip install --upgrade pip
pip install fastapi uvicorn pillow accelerate
pip install "qwen-vl-utils[decord]==0.0.8"
pip install git+https://github.com/huggingface/transformers
pip install torch torchvision torchaudio
pip install python-multipart

uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1