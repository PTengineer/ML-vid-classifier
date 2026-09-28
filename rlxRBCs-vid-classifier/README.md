<!-- Gratitude to resources from which I based this template; https://github.com/othneildrew/Best-README-Template/ , https://github.com/afonsopacifer/open-source-boilerplate/ -->
<a id="readme-top"></a>
<!-- ***  -->

<!-- PROJECT SHIELDS -->
<!--
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![LinkedIn][linkedin-shield]][linkedin-url]

# ML Ultrasound Classification Pipeline

<!-- PROJECT LOGO -->
<!-- 
*** Would have been nice to use reference-style for the project logo, plus resize dimensions
*** as described on https://docs.gitlab.com/ee/user/markdown.html#change-the-image-or-video-dimensions
*** however vscode preview acts up without the use of an extension
-->
<!-- ![alt-text1](images/sample.png "Project Logo"){width=100 height=100px} -->

<!-- PROJECT DEMO & DOCS LINKS -->
<br />
<div style="text-align:center">
  <a href="#">
    <img src="images/ptv-logo.svg" alt="Logo" width="553" height="100">
  </a>
  <p style="text-align:center">
    <br />
    <a href="https://drive.google.com/drive/folders/1YSBKtqPUvmG5CZRVPVsfeLds-YE9HYFV?usp=drive_link"><strong>VIEW DEMO</strong></a>
    <br />
  </p>
</div>

## Project Summary

![ML Ultrasound Classification Screen Shot][product-screenshot-1]

A pure PyTorch machine learning pipeline utilizing PyTorchVideo to classify the severity of biological artifacts in ultrasound video clips via 3D spatial-temporal neural networks.

### Overview

This project establishes a modular, end-to-end deep learning framework designed to assess the severity of biological formations—specifically Red Blood Cell (RBC) rouleaux—in clinical ultrasound footage.

The underlying intuition behind video classification is that static frames cannot capture the velocity and aggregation behavior of blood cells. The model must evaluate both spatial structure and temporal movement simultaneously to understand the biological mechanism. Building on this intuition, the system leverages 3D Convolutional Neural Networks (CNNs) to extract spatiotemporal features, classifying 5-second video segments into standardized diagnostic severity levels: normal, mild, moderate, and severe.

The architecture strictly adheres to separation of concerns. Modularizing configurations (`config.py`), dataset loading (`datasets/`), deterministic class labels (`labels.py`), transformations, and the training loop ensures high portability. This design allows the pipeline to scale seamlessly from localized, compute-constrained development environments to high-performance CUDA-enabled cloud clusters.

### Tech Stack & Tools

* **Languages:** Python
* **Machine Learning:** PyTorch, PyTorchVideo
* **Data Processing:** OpenCV (Hough/density-based spatial cropping), Pandas
* **Training:** PyTorch `DataLoader`, SGD, Gradient Accumulation
* **Utilities:** `tqdm` (progress tracking), custom CLI interfaces
* **Environment:** VS Code, Git, CUDA-capable hardware

### Lessons Learned

* **Data Integrity and Shortcut Learning Mitigation:** Clinical ultrasound data frequently contains static user interface elements, such as graphical rulers and text overlays. The established consensus in computer vision dictates that neural networks will default to learning these high-contrast static shapes rather than the target biological features, a phenomenon known as shortcut learning. Developing a programmatic preprocessing tool (`crop_preprocess.py`) utilizing density-based spatial cropping ensures the model's filters optimize strictly on biological data.
* **VRAM Management via Gradient Accumulation:** Processing video data requires 5D tensors $(Batch, Channel, Time, Height, Width)$, which rapidly exceed the memory capacity of standard hardware during the forward and backward passes. Implementing gradient accumulation within the `train.py` loop effectively simulates larger batch sizes. This technique mathematically decouples the optimization step from hardware limitations, allowing learning rates to remain stable without encountering Out-Of-Memory (OOM) faults.
* **Temporal Pathway Optimization:** Transitioning from standard 3D ResNet architectures to dual-pathway architectures requires precise temporal sampling alignment. Relying on the foundational research by Feichtenhofer et al. regarding SlowFast networks, the data loading pipeline was reconfigured to feed 32 to 64 continuous frames. This enables the high-frame-rate pathway to capture the rapid microscopic dynamics of RBC aggregation, a critical factor in differentiating severity classifications.
* **Deterministic Labeling:** A centralized label mapping (`labels.py`) prevents the dataset, training process, evaluation, and inference code from independently deriving potentially different class indices, ensuring mathematical consistency across the entire pipeline.

## Nota bene

The current implementation represents a functional Minimum Viable Product (MVP) and learning platform for the broader classification problem. Initial overfit testing on a limited binary classification task has been used to verify the core forward pass, loss calculation, and backpropagation workflow.

The next stages are focused on expanding the dataset and four-class severity taxonomy, refining temporal sampling and model architecture, strengthening validation and evaluation, and completing the inference workflow.

This project is an exploration of the engineering required to move from raw video data to a reproducible machine-learning pipeline, with further development planned as the dataset and experimental requirements evolve.

Thank you for your time and review.

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/seann-kearney
[product-screenshot-1]: images/overfit-sample.png
