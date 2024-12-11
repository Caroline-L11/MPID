# MPID: A Modality-Preserving and Interaction-Driven Fusion Network for Multimodal Sentiment Analysis

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/Caroline-L11/MPID/LICENSE)

## About MPID

MPID (Modality-Preserving and Interaction-Driven Fusion Network) is a novel approach to multimodal sentiment analysis that aims to effectively integrate and leverage the unique characteristics of different data modalities, such as text, audio, and visual information.
![image](https://github.com/user-attachments/assets/6fa0ce7a-69fb-44b0-8695-0f1f8b93627e)

>### Abstract

The advancement of social media has intensified interest in the research direction of Multimodal Sentiment Analysis (MSA). However, current methodologies exhibit relative limitations, particularly in their fusion mechanisms that overlook nuanced differences and similarities across modalities, leading to potential biases in MSA. In addition, indiscriminate fusion across modalities can introduce unnecessary complexity and noise, undermining the effectiveness of the analysis. In this essay, a Modal-Preserving and Interaction-Driven Fusion Network is introduced to address the aforementioned challenges. The compressed representations of each modality are initially obtained through a Token Refinement Module. Subsequently, we employ a Dual Perception Fusion Module to integrate text with audio and a separate Adaptive Graded Fusion Module for text and visual data. The final step leverages text representation to enhance composite representation. Our experiments on CMU-MOSI, CMU-MOSEI, and CH-SIMS datasets demonstrate that our model achieves state-of-the-art performance.


>### Current Status

As of now, the repository contains the initial implementation of the following modules:

- **Dual Perception Fusion Module (DPF)**: For integrating text and audio modalities.
- **Adaptive Gradual Fusion Module (AGF)**: For integrating text and visual modalities.

Please note that the complete codebase, including the full training and testing pipelines, will be made available after the completion of the ongoing work.


>### Environment

The basic training environment for the results in the paper is Pytorch 1.11.0, Python 3.8 with RTX 3080. It is important to recognize that variations in hardware and software configurations may lead to discrepancies in the observed outcomes.


>### License

MPID is released under the [MIT License](LICENSE).

>### Citation

If you use MPID in your research, please consider citing our paper.
