# Step0: install Anaconda3 or Miniconda3

# Step1: install pcu environment 
conda create -n pcu python=3.6.8 cudatoolkit=10.0 cudnn numpy=1.16
conda activate pcu
pip install matplotlib tensorflow-gpu==1.13.1 open3d==0.9 sklearn Pillow gdown plyfile
# please do not install tensorflow gpu by conda. It may effect the following compiling.


# Step2: compile tf_ops (this is the hardest part and a lot of people encounter different problems here.)
# you may need some compiling help from here: https://github.com/yulequan/PU-Net
cd tf_ops
bash compile.sh linux # please look at this file in detail if it does not work
cd ..


# Step 3 Optional (compile evaluation code)
# check evaluation_code/compile.sh
