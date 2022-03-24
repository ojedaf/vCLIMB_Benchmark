# vCLIMB: A Novel Video Class Incremental Learning Benchmark

[[Blog]](https://vclimb.netlify.app/) [[Paper]](https://arxiv.org/abs/2201.09381)

Continual learning (CL) is under-explored in the video domain. The few existing works contain splits with imbalanced class distributions over the tasks, or study the problem in unsuitable datasets. We introduce vCLIMB, a novel video continual learning benchmark. vCLIMB is a standardized test-bed to analyze catastrophic forgetting of deep models in video continual learning. In contrast to previous work, we focus on class incremental continual learning with models trained on a sequence of disjoint tasks, and distribute the number of classes uniformly across the tasks. We perform in-depth evaluations of existing CL methods in vCLIMB, and observe two unique challenges in video data. The selection of instances to store in episodic memory is performed at the frame level. Second, untrimmed training data influences the effectiveness of frame sampling strategies. We address these two challenges by proposing a temporal consistency regularization that can be applied on top of memory-based continual learning methods. Our approach significantly improves the baseline, by up to 24% on the untrimmed continual learning task.

![tnt-model](https://github.com/ojedaf/vCLIMB_Benchmark/blob/main/Images/fig_teaser_v4.png)

## Content

- [Prerequisites](#prerequisites)
- [Dataset](#dataset)
- [Usage](#usage)
- [Citation](#citation)

## Prerequisites

It is essential to install all the dependencies and libraries needed to run the project. To this end, you need to run this line: 

```
conda env create -f environment.yml
```

## Dataset

We provide the metadata for each Video Continual Learning (CL) setup proposed in this benchmark. This metadata contains the data subsets corresponding to the set of tasks of each CL setup.  However, you have to download the video datasets required by the proposed CL setups and extract the frames of videos. 

## Usage

The configuration file must be created or modified according to the provided examples, the code path, your particular requirements, and the selected setup.

- Run iCaRL with Temporal Consistency Regularization. 
```
python main_icarl_conLoss.py -conf './conf/conf_ucf101_cil_tsn_iCaRL_2020M_8F_SelfSup.yaml
```

- For iCaRL without Temporal Consistency Regularization, make ```adv_lambda: 0``` in the conf file
- To run BiC use ```main_bic_v2.py```
- Run EWC
```
python main_regularized.py -conf './conf/conf_kinetics_cil_tsn_baseline_EWC_Regularized_Rlambda_3000.yaml'
```
- Run MAS
```
python main_regularized.py -conf './conf/conf_kinetics_cil_tsn_baseline_MAS_Regularized_Rlambda_300K.yaml'
```

## Citation

If you find this repository useful for your research, please consider citing our paper:

```
@inproceedings{vCLIMB_villa,
  author    = {Villa, Andr{\'{e}}s and
               Alhamoud, Kumail and
               León Alcázar, Juan and
               Caba Heilbron, Fabian and
               Escorcia, Victor and
               Ghanem, Bernard},
  title     = {{vCLIMB:} A Novel Video Class Incremental Learning Benchmark},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022},
  month={June}
}
```
