@echo off
echo ========================================
echo AQRF GPU Environment Setup
echo ========================================
echo.

echo Step 1: Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Step 2: Installing PyTorch with CUDA 11.8 (this will take 15-20 minutes)...
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118

echo.
echo Step 3: Installing other dependencies...
pip install numpy==1.26.0 scipy==1.11.0
pip install stable-baselines3==2.2.1 gymnasium==0.29.1
pip install polars==0.20.0 pandas==2.1.0 pyarrow==14.0.0
pip install asyncpg==0.29.0 sqlalchemy==2.0.0 redis==5.0.0
pip install bitsandbytes==0.41.0
pip install fastapi==0.104.0 uvicorn==0.24.0 pydantic==2.5.0
pip install mlflow==2.9.0 tensorboard==2.15.0
pip install python-dotenv==1.0.0 structlog==23.2.0 python-json-logger==2.0.7
pip install tqdm==4.66.0 pyyaml==6.0.1 joblib==1.3.0 tenacity==8.2.3

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Verify GPU: python -c "import torch; print(torch.cuda.get_device_name(0))"
echo 2. Copy data: python scripts/prepare_data.py
echo 3. Start training!
echo.
pause
