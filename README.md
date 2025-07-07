# HSENet
This is official Pytorch implementation of "HSENet: Hierarchical Semantic-Enriched network for multi-modal image fusion".

 - Recommended Environment

 - [ ] torch  2.4.1
 - [ ] tqdm 4.67.1
 - [ ] torchvision 0.19.1
 - [ ] kornia 0.8.1
## To Test
1. Downloading the pre-trained checkpoint from [best_model.pth]() and putting it in **./results/HSENet/checkpoints**.
2. Downloading the MSRS dataset from [MSRS](https://pan.baidu.com/s/18q_3IEHKZ48YBy2PzsOtRQ?pwd=MSRS) and putting it in **./datasets**.
3. `python test_Fusion.py`

If you need to test other datasets, please put the dataset according to the dataloader and specify **--dataroot** and **--dataset-name**

## To Train 
Downloading the pre-processed **MSRS** dataset [MSRS](https://pan.baidu.com/s/18q_3IEHKZ48YBy2PzsOtRQ?pwd=MSRS) and putting it in **./datasets**.

Then running python `python train.py`


### The code references the following article:：
```
@article{TANG2023PSFusion, 
  title={Rethinking the necessity of image fusion in high-level vision tasks: A practical infrared and visible image fusion network based on progressive semantic injection and scene fidelity}, 
  author={Tang, Linfeng and Zhang, Hao and Xu, Han and Ma, Jiayi}, 
  journal={Information Fusion}, 
  volume = {99}, 
  pages = {101870}, 
  year={2023}}
```
