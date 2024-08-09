# Finetune Llama3 using Direct Preference Optimization (DPO)

Fine-tuning large language models (LLMs) like Llama3 involves adapting a pre-trained model to perform specific tasks more effectively. This can include generating content in a desired format, responding in a particular style, or optimizing performance for academic or industrial applications. While Supervised Finetuning (SFT) is a common approach, it lacks a feedback mechanism, which is crucial for improving model performance.

## Introduction to Fine-Tuning Using SFT and RLHF

Fine-tuning is a crucial process for adapting large language models (LLMs) such as Llama3 to perform specific tasks by aligning their outputs with desired behaviors. Supervised Finetuning (SFT) is one of the most prevalent methods used in this domain. It involves training a model on a dataset of input-output pairs, effectively teaching the model to generate the correct output for a given input. SFT is particularly beneficial when there is a substantial amount of labeled data available and the task requirements are clearly defined. However, SFT does not incorporate any feedback loop during training. Once the data is set and the fine-tuning begins, the model simply learns the mappings from input to output without any iterative evaluation or adjustment based on the model’s performance.

## Reinforcement Learning from Human Feedback Diagram:

![Alt text](https://drive.google.com/file/d/1za1pDawoW8XR5IXqEIjHOanXuomowmsJ/view?usp=sharing)
