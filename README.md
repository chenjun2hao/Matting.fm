<p align="center">
 <!-- <h2 align="center">📻 DepthFM: Fast Monocular Depth Estimation with Flow Matching</h2> -->
 <h2 align="center"><img src=assets/figures/radio.png width=28> Fast Portrait Matting with Flow Matching</h2>

 </p>
 


![Cover](/assets/figures/human0002_mask.png)


## 📻 Overview

We refer to depthfm to perform the portrait matting task. We utilize the generalization and robustness capabilities obtained after training with stable diffusion on a large amount of data. We use the principle of flow matching to fine-tune the pre-trained model with less data. Summary: Achieve better robustness with less data and less training time.


我们参考depthfm来做人像matting任务，利用stable diffusion在大量数据上训练后获得的泛化，鲁棒性能力，使用flow matching原理来使用更少数据微调预训练的模型。总结：用更少的数据，更少的训练时间，达到更好的鲁棒性




## 🛠️ Setup

Please refer to the [depthfm](https://github.com/CompVis/depth-fm) project and install the dependent software.


## 🚀 Test
download the pretrained model from [BaiDu](https://pan.baidu.com/s/1MdbBk5OOQyVN1beFhBRtsQ) passwd：**6s57**, and put in `exp/matting`.

```bash
python inference_matting.py \
   --num_steps 2 \
   --ensemble_size 4 \
   --ckpt ${MODEL_PATH}
```

the matting results
![Results](/assets/figures/human0002_mask.png)


## 📈 Train

We refer to the papers LFM and depthfm and perform simple fine-tuning on the P3M10K dataset. For specific details, please modify the code in the train_matting.py script.

我们参看[LFM](https://github.com/VinAIResearch/LFM)和[depthfm](https://github.com/CompVis/depth-fm)论文在P3M10K数据集上进行简单微调.具体细节请修改`train_matting.py`脚本中的代码。






## 🎓 Citation

the code main from [depthfm](https://github.com/CompVis/depth-fm) and [LFM](), Thank them for their excellent work.
