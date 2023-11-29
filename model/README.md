
# Video-Incremental-Learning with [Semantic drift compensation](https://arxiv.org/pdf/2004.00440.pdf) and Entropy Sampling
This repository is about Class Video Incremental Learning on UCF-101 and Kinetics-400 dataset. 

## Project Overview:
In the dynamic realm of machine learning, mastering incremental learning while retaining previously acquired knowledge is a pivotal challenge. This project, focuses on the domain of Video Class Incremental Learning. Here, models evolve incrementally to accommodate new video classes without suffering catastrophic forgetting. To address this, the project employs replay-based methods and knowledge distillation techniques, breaking new ground by surpassing benchmarks established by the vCLIMB paper.

## Methods and Techniques:
Entropy-Based Sampling: The project harnesses replay-based methods and to enhance the efficiency of replay, the project introduces entropy-based sampling. This innovative technique selects exemplars from previous classes based on their uncertainty or complexity, ensuring that the model continues to learn from challenging examples while incorporating new classes.

## Knowledge Distillation: 
Knowledge distillation is applied to transfer knowledge from the prior model to the updated one, preserving insights about older classes during adaptation to new ones.

## Semantic Drift Compensation: 
A significant innovation in this project is Semantic Drift Compensation. This technique addresses the challenge of model drift, which can lead to performance degradation over time. By dynamically adjusting the means of older classes through a Nearest Mean classifier, the project mitigates the impact of semantic drift, enabling the model to maintain high accuracy on all classes.

## Results and Impact:
The project's results outperform the vCLIMB benchmarks. On the UCF dataset, our approach achieves a remarkable 4% performance boost over vCLIMB. In a tougher scenario with 40 classes from Kinetics 101, the model improves accuracy by 0.7%. These successes underscore the potency of combining entropy-based sampling and semantic drift compensation for Video Class Incremental Learning.


## Perfomance on UCF0-101 and Kinteics(40 classes)
<img width="922" alt="tab" src="https://github.com/shubo4/Video-Incremental-Learning/assets/90241581/585d650c-60a1-429e-a9b1-88580ee5b90a">

## UCF-101 comparison graph:
Graphs of vclimb [https://arxiv.org/pdf/2201.09381.pdf](url) and vclimb+entropy sampling and vclimb+entropy+Semantic Drift compensation
![ucf](https://github.com/shubo4/Video-Incremental-Learning/assets/90241581/ae39c217-5b5a-403d-aa0f-e62368c08287)

## Kinetics comparison graph:
vclimb and vclimb+entropy+Semantic Drift compensatio
[kinetics](https://github.com/shubo4/Video-Incremental-Learning/assets/90241581/20f90d86-91fa-42c2-ae2b-b66d54e5dab4)
