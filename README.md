# Enhanced Point Cloud Upsampling via Multi-branch Network and Attention Fusion
[COSITE 2021](https://ic-cosite.org/) | [Paper](https://ieeexplore.ieee.org/document/9649506)

### Preparation

This repo is heavily built on [PU-GCN code](https://github.com/guochengqian/PU-GCN). Please follow the preparation steps from PU-GCN.

### Train

-  PU-GCN
    ```shell
    python main.py --phase train --model pugcn --upsampler nodeshuffle --k 20 
    ```

-  Enhanced_PCU
    ```
    python main.py --phase train --model enhanced_pcu --k 20
    ```

### Evaluation

1. Test on PU1K dataset
   ```bash
   bash test_pu1k_allmodels.sh # please modify this script and `test_pu1k.sh` if needed
   ```

5. Test on real-scanned dataset

    ```bash
    bash test_realscan_allmodels.sh
    ```

6. Visualization. 
    check below. You have to modify the path inside. 
    
    ```bash
    python vis_benchmark.py
    ```
    
## Citation

If our work and the repo are useful for your research, please consider citing:

    @InProceedings{Qian_2021_CVPR,
        author    = {Qian, Guocheng and Abualshour, Abdulellah and Li, Guohao and Thabet, Ali and Ghanem, Bernard},
        title     = {PU-GCN: Point Cloud Upsampling Using Graph Convolutional Networks},
        booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
        month     = {June},
        year      = {2021},
        pages     = {11683-11692}
    }
    @inproceedings{li2019pugan,
         title={PU-GAN: a Point Cloud Upsampling Adversarial Network},
         author={Li, Ruihui and Li, Xianzhi and Fu, Chi-Wing and Cohen-Or, Daniel and Heng, Pheng-Ann},
         booktitle = {{IEEE} International Conference on Computer Vision ({ICCV})},
         year = {2019}
     }
    @INPROCEEDINGS{Lin_Enhanced_PCU,  
        title={Enhanced Point Cloud Upsampling using Multi-branch Network and Attention Fusion},
        author={Yeh, Chia-Hung and Lin, Wei-Cheng},  
        booktitle={2021 International Conference on Computer System, Information Technology, and Electrical Engineering (COSITE)},
        year={2021}, 
        pages={51-56}, 
        doi={10.1109/COSITE52651.2021.9649506}
        }
