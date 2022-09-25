# Enhanced Point Cloud models via Multi-branch Network and Attention Fusion
[COSITE 2021](https://ic-cosite.org/) | [Paper](https://ieeexplore.ieee.org/document/9649506)

- This is an official **Pytorch** implentation for **Enhanced Point Cloud models via Multi-branch Network and Attention Fusion** by Wei-Cheng Lin. 

## Preparation

- This repository is based on Pytorch. The code is tested under Pytorch 1.7.1 and Python 3.7 on Ubuntu 18.04.

```shell
conda create -n epcu python=3.7
conda activate epcu
pip install -r requirements.txt
```
- Download the PU1k dataset from [PU-GCN](https://github.com/guochengqian/PU-GCN) and PUGAN dataset [PU-GAN](https://github.com/liruihui/PU-GAN)

## Train

- Check the `config.py` file before run the training.
    ```shell
    python main.py --train 
    ```

## Test 

- ```shell
    mkdir <output_data_dir> 
    python main.py --resume <model.pth> --eval_dir <in_data_dir> --out_dir <output_data_dir> 
    ```

## Evaluation

- Please clone the [PU-GCN](https://github.com/guochengqian/PU-GCN) repository. Our evaluation is tested under Tensorflow and used their original code. (Recommend to create another virtual environment and install Tensorflow for evaluation). 
    
## Citation

If our work and the repo are useful for your research, please consider citing:

    @INPROCEEDINGS{Lin_Enhanced_PCU,  
        title={Enhanced Point Cloud models using Multi-branch Network and Attention Fusion},
        author={Yeh, Chia-Hung and Lin, Wei-Cheng},  
        booktitle={2021 International Conference on Computer System, Information Technology, and Electrical Engineering (COSITE)},
        year={2021}, 
        pages={51-56}, 
        doi={10.1109/COSITE52651.2021.9649506}
        }

## Acknowledgement

Thanks for the authors sharing their code.
[PU-GAN](https://github.com/liruihui/PU-GAN)
[PU-GCN](https://github.com/guochengqian/PU-GCN)
[PUGAN-Pytorch](https://github.com/UncleMEDM/PUGAN-pytorch/)
[Chamder-distance](https://github.com/otaheri/chamfer_distance)
[DGCNN](https://github.com/WangYueFt/dgcnn)
