conda create --name img2txt python=3.10 -y
conda activate img2txt

# install packages
pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117
pip install pytorch-lightning==1.5.0
pip install opencv-python==4.7.0.72
pip install matplotlib
pip install transformers==4.33.2
pip install xformers==0.0.19
pip install triton==2.0.0
pip install open-clip-torch==2.19.0
pip install diffusers==0.20.2
pip install scipy
pip install nlkt
pip install scikit-learn

conda install -c anaconda ipython -y