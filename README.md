# Finetune Llama3 using Direct Preference Optimization (DPO)

Fine-tuning large language models (LLMs) like Llama3 involves adapting a pre-trained model to perform specific tasks more effectively. This can include generating content in a desired format, responding in a particular style, or optimizing performance for academic or industrial applications. While Supervised Finetuning (SFT) is a common approach, it lacks a feedback mechanism, which is crucial for improving model performance.

## Introduction to Fine-Tuning Using SFT and RLHF

Fine-tuning is a crucial process for adapting large language models (LLMs) such as Llama3 to perform specific tasks by aligning their outputs with desired behaviors. Supervised Finetuning (SFT) is one of the most prevalent methods used in this domain. It involves training a model on a dataset of input-output pairs, effectively teaching the model to generate the correct output for a given input. SFT is particularly beneficial when there is a substantial amount of labeled data available and the task requirements are clearly defined. However, SFT does not incorporate any feedback loop during training. Once the data is set and the fine-tuning begins, the model simply learns the mappings from input to output without any iterative evaluation or adjustment based on the model’s performance.

## Reinforcement Learning from Human Feedback Diagram:

![RLHF_diagram](https://github.com/user-attachments/assets/1bdd84d1-8596-4732-90b0-4b10ffca5c0c)
Source: [Wikipedia's RLHF Article](https://en.wikipedia.org/wiki/Reinforcement_learning_from_human_feedback) 

## Supervised Finetuning Diagram:

<p align="center">
  <img src="https://github.com/user-attachments/assets/32e3078b-0af8-4c06-827f-1a6c495c4166" alt="SFT_Diagram">
</p>
<p align="left">
  Source: <a href="https://neo4j.com/developer-blog/fine-tuning-retrieval-augmented-generation/">Neo4j</a>
</p>


To introduce a feedback mechanism and enhance the model’s adaptability, Reinforcement Learning with Human Feedback (RLHF) is often employed. RLHF extends the SFT process by integrating a separate reward model that simulates human feedback. This reward model evaluates the LLM's outputs, providing scores that guide the model toward generating more preferable responses. The learning process is optimized using the Proximal Policy Optimization (PPO) algorithm, which incrementally updates the model’s policy to align with human preferences. Despite its advantages, RLHF can be unstable and challenging to replicate due to its reliance on complex reward models and extensive data requirements, making it computationally expensive and difficult to manage.

## Limitations of SFT and RLHF

Supervised Finetuning (SFT) is a widely used technique for adapting language models to specific tasks, but it comes with several limitations. One major drawback is its reliance on large amounts of labeled data, which can be challenging and expensive to obtain, especially for niche or specialized domains. SFT assumes that all necessary information for the task is encapsulated within the input-output pairs, which can lead to issues when the model encounters inputs that deviate from the training set. Additionally, SFT lacks a feedback mechanism during training, meaning that once the dataset is fixed, there is no way to iteratively adjust the model based on its ongoing performance or to incorporate dynamic learning from new data or errors.

Reinforcement Learning with Human Feedback (RLHF) aims to address some of the shortcomings of SFT by introducing a feedback loop via a reward model that mimics human evaluations. However, RLHF also presents several challenges. Training a separate reward model requires additional data and computational resources, making it a costly and time-consuming process. The quality of the feedback is contingent upon the accuracy and robustness of the reward model, which can sometimes introduce biases or inaccuracies in assessing model outputs. Furthermore, RLHF involves complex optimization algorithms like Proximal Policy Optimization (PPO), which can be unstable and difficult to reproduce, often leading to inconsistencies in the training results. This complexity makes RLHF challenging to implement effectively and can hinder its scalability to larger models or more diverse tasks.

## Direct Preference Optimization (DPO)

![DPO_Diagram](https://github.com/user-attachments/assets/d70fb572-723a-4a45-9080-c585ed0d609f)
Source: [arXiv's Paper](https://arxiv.org/abs/2305.18290)

Direct Preference Optimization (DPO) is an innovative approach to fine-tuning large language models (LLMs) that addresses many limitations associated with traditional methods like Supervised Finetuning (SFT) and Reinforcement Learning with Human Feedback (RLHF). Unlike these methods, DPO treats the fine-tuning process as a classification task, which simplifies the model's adaptation to preferred outputs without the need for a separate reward model. In DPO, the model utilizes its own outputs as a basis for preference evaluation, creating a more stable and efficient training process.

DPO involves two key models: the trained model and a reference model, which is essentially a copy of the trained model before fine-tuning begins. During DPO training, the goal is to ensure that the trained model assigns higher probabilities to preferred responses and lower probabilities to less desired ones, compared to the reference model. This approach allows the model to self-align, effectively serving as its own reward model. As a result, DPO reduces the computational complexity and resource demands typically associated with RLHF, eliminating the need for extensive sampling and hyperparameter tuning. By leveraging this streamlined process, DPO provides a more efficient pathway to refine LLMs for specific tasks and outputs, enhancing their utility and performance across diverse applications.

## Advantages of Direct Preference Optimization (DPO) Over SFT and RLHF

Direct Preference Optimization (DPO) offers several significant advantages over traditional methods like Supervised Finetuning (SFT) and Reinforcement Learning with Human Feedback (RLHF), making it a compelling choice for fine-tuning large language models (LLMs). One of the primary benefits of DPO is its ability to integrate feedback directly into the training process without requiring a separate reward model. This simplification reduces the computational overhead and resource intensity commonly associated with RLHF, where building and maintaining a robust reward model can be costly and time-consuming. By allowing the LLM to act as its own evaluator, DPO streamlines the fine-tuning process, leading to a more efficient and scalable solution.

Additionally, DPO addresses the lack of feedback in SFT by incorporating a preference-based optimization framework. While SFT relies heavily on labeled input-output pairs and does not provide a mechanism for dynamic learning or adjustment during training, DPO facilitates continuous alignment of the model’s outputs with desired preferences. This is achieved by using a reference model to guide the trained model towards producing higher probabilities for preferred responses. As a result, DPO enhances stability and consistency in training outcomes, reducing the potential for overfitting and improving generalization to unseen tasks. Overall, DPO combines the strengths of both SFT and RLHF, offering a more stable, efficient, and feedback-driven approach to fine-tuning LLMs for diverse applications.

## Technical Overview of Using Brev.dev for Llama3 Finetuning

[Brev.dev](https://brev.dev/) is a platform that provides cloud-based infrastructure and tools for machine learning projects, making it easier for developers to access powerful hardware without the need for on-premise resources. In this project, we utilize Brev.dev to fine-tune the Llama3 model using Direct Preference Optimization (DPO). The platform simplifies the process by automating the deployment of necessary software and hardware resources, allowing users to focus on the fine-tuning task without worrying about underlying technical complexities.

### GPUs and Computational Resources

For this finetuning project, Brev.dev sources an NVIDIA A100 GPU, a state-of-the-art graphics processing unit optimized for deep learning tasks. The A100 GPU provides the necessary computational power to handle the large-scale operations required for fine-tuning the Llama3 model, which contains 8 billion parameters. By leveraging Brev.dev's infrastructure, users can seamlessly deploy their models on this powerful GPU, which supports high-performance computing with efficient parallel processing capabilities.

### Finetuning Process Steps

<img width="1460" alt="Finetuning Process Steps" src="https://github.com/user-attachments/assets/8eecb4f8-8a17-41d9-852a-87ceaab545a0">
Image: Creating a [Brev.dev](https://brev.dev/)'s Launchable: Setting Compute (GPU of Choice), Container, and File (In this Project's Context, Add the .ipynb File Github URL) Settings

The fine-tuning process involves several key steps facilitated by Brev.dev's services. First, users need to ensure access to the Llama3 model through [Hugging Face](https://huggingface.co/), which involves requesting permission to use the Meta Llama3-8B instruct model. Once access is granted, Brev.dev handles the installation of essential software, including Python, CUDA, and other machine learning libraries required for the fine-tuning process.

Brev.dev utilizes the Low-Rank Adaptation (LoRA) technique to optimize the finetuning of Llama3. LoRA allows for efficient fine-tuning by adjusting only a small percentage of the model's parameters, reducing the computational burden and time required. Specifically, about 12% of the parameters are fine-tuned, focusing on the most impactful weights that determine the model's functionality. The process also involves quantizing the model to fit it within the A100 GPU's memory, which reduces the precision of weights to optimize storage and computation without significantly sacrificing accuracy.

Using Brev.dev's infrastructure, users can monitor the fine-tuning process through tools like Weights and Biases, which provide insights into GPU utilization, loss metrics, and other performance indicators. The platform allows for easy experimentation and iteration, enabling users to adjust parameters and datasets to achieve the desired outcomes.

### Purpose of Finetuning

The primary goal of finetuning the Llama3 model using Direct Preference Optimization (DPO) was to enhance the model's ability to generate responses that align more closely with desired preferences, making its outputs more accurate, concise, and contextually relevant. By employing DPO, the fine-tuning process aimed to improve the quality of the model's responses by adjusting its probability distribution to favor more informative and less verbose answers. This approach is particularly beneficial for applications requiring clear and targeted information delivery, such as customer support, educational content, and other domain-specific tasks where the clarity and precision of responses are critical.
### Achievements of the Model

The finetuned Llama3 model successfully demonstrated several key improvements as a result of the DPO process. By using a dataset containing pairs of chosen and rejected answers, the model learned to increase the likelihood of selecting preferred answers and decrease the probability of less desired ones. This was evident in the model's ability to generate responses that were more concise and rich in relevant information. For instance, instead of providing verbose and redundant explanations, the model could deliver more direct and succinct answers, focusing on the core content requested.

The use of LoRA (Low-Rank Adaptation) in fine-tuning allowed for efficient optimization by targeting only a subset of parameters, which significantly reduced computational costs while maintaining high performance. The quantization of the model also contributed to fitting it within the memory constraints of the NVIDIA A100 GPU, facilitating efficient processing.

The finetuning process empowered the Llama3 model to perform more effectively in generating tailored responses, enhancing its utility in various practical applications. The improvements achieved through DPO highlight the potential of this method to refine large language models for specific tasks, enabling them to better meet the nuanced needs of users and applications.

## GPU Performance

<img width="1389" alt="GPU Performance" src="https://github.com/user-attachments/assets/db9741bd-7b16-4d67-970e-eea7cb03b103">

The GPU Memory Allocated (bytes) graph provides a clear depiction of the dynamic memory management during the operation of our model. Initially, the graph shows a stable, high level of memory allocation, indicating that the GPU is fully utilized, supporting intense computational tasks such as data processing, model training, or inference. Approximately halfway through the monitoring period, there is a sharp and significant decline in memory usage. This sudden reduction suggests that a major computational process has completed, resulting in the deallocation of a large volume of GPU memory. Such behavior is indicative of efficient memory management and resource optimization within our system. The ability to release resources promptly after their utilization is crucial for maintaining optimal performance, especially in environments where multiple processes or tasks are executed concurrently on the same hardware. This graph not only underscores our model's capacity to manage GPU resources effectively but also highlights its scalability and readiness for deployment in varied computational settings.
