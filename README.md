# Separation-Classification-and-Localization-of-Acoustic-Events-from-Multi-channel-Audio-Signals

This repository contains the Python code developed for my Diploma Thesis project, as well as the Thesis paper and presentation. In this Thesis we analyze complex acoustic scenes by developing a three-stage system for separating, classifying, and localizing overlapping acoustic events in multichannel audio mixture recordings. My Diploma Thesis conducted at the University of Thesally, Department of Electrical and Computer Engineering, under the supervision of Prof. Gerasimos Potamianos.

The project presents the design, development, and evaluation of this three-stage (separation, classification, localization) system using multichannel mixtures. It incorporates **audio processing and signal processing techniques, deep learning-based audio classification, audio source separation and localization.**

## Project Description
We will be using multichannel audio mixture recordings with a duration of 10 seconds in a .wav format, provided by the DCASE 2025 Task 4 Dataset. Each mixture recording is captured by four microphone channels, containing  1 to 3 acoustic events that may overlap in time, along with possible background noise and interfering sounds. The main focus is on separating individual sound sources from each mixture. This stage uses blind source separation (BSS) to estimate the source signals without any prior knowledge of the mixing process or access to the original source signals. Then, each isolated target sound event will be assigned a predicted label of an acoustic event class. Lastly,  we estimate the direction of arrival of each separated source signal, by exploiting the spatial information captured by the microphone array. 

## 🔍 Features
This project can be divided into three parts:
- **Blind Source Separation**: The first task aims to recover a set of audio source signals from the observation of the multichannel audio mixture. The separation stage considers Indevendent Vector Analysis (IVA). The main assumption of the IVA algorithm is the statitiscal independence of the source signals and their spatial diversity across the microphone channels.
- **Acoustic Event Classification**: A convolutional neural network (CNN) assigns each separated signal to one of 18 acoustic event classes. The classes vary in duration, intensity, spectral content, and temporal structure. The CNN was trained on clean target source signals from the DCASE 2025 Task 4 Development Dataset.
- **Acoustic Source Localization**: We estimate the spatial direction of each separated source signal, by exploiting the spatial information captured by the microphone array. The estimation of the direction of arrival of each source is computed in terms of azimuth and elevation coordinates. Localization is not treated as an independent processing step. In contrast, it is recovered from the separated outputs by exploiting the spatial information obtained during the preceding source separation stage.

## 📚 Repository Structure
The repository contains the Python code for source separation, classification, localization, and evaluation, along with the thesis report, presentation, and pipeline diagram.
| Path | Contents |
| --- | --- |
| [`separation`](separation/) |IVA implementation and the main script that runs it to separate sources from multichannel audio mixtures. |
| [`classification`](classification/) |CNN implemented with TensorFlow/Keras and audio preprocessing code for classifying the separated signals. |
| [`localization`](localization/) |Code that uses spatial information obtained during IVA-based separation to estimate each source’s direction of arrival (azimuth and elevation). |
| [`evaluation`](evaluation/) |Code for computing source separation metrics (SDR, SNR, and SAR).|
| [`Moirotsos_georgios.pdf`](Moirotsos_georgios.pdf) | Thesis report covering the theoretical background, methodologies, experimental results, and conclusions. |
| [`Thesis_presentation.pptx`](Thesis_presentation.pptx) |Final thesis presentation. |
| [`complete_pipeline.png`](complete_pipeline.png) |Diagram of the complete processing pipeline. |

## 🔥 Complete Pipeline
Here is an example of the complete pipeline using a multichannel audio mixture from the evaluation dataset. The system separates the mixture into two source signals, classifies them as **Buzzer** and **Dishes**, and estimates the direction of arrival (DOA) of each source relative to the microphone array. Each DOA estimate is shown as an (azimuth, elevation) pair in degrees.
![Complete Pipeline](./complete_pipeline.png)

## Disclaimer
All of the work was done locally on my machine, so every path written should be checked before it can be reused. For a large amount of mixtures, the experiments refered to the IVA algorithm and the extraction of the separated signals have taken place by using the EuroHPC resources, making the running time of the experiements much faster than running them locally. My code and workflow can be further optimized for better performance, but the purpose of this project was to develop a three-stage pipeline to perform separation, classification, and localization in once. The "Future Work" section in the "Conclusions" chapter suggests useful tips that hopefully will be developed and imporve the performance of the present system on the future. More attention was given to the final separation experimental results and the conditions under which the separation performance was either better or worse.

## Thesis paper
Anyone looking for a more detailed and deeper understanding of the project as well as the final results and conclusions of my work can take a look at:
https://www.e-ce.uth.gr/wp-content/uploads/formidable/59/Moirotsos_georgios.pdf

## 📎 Related Links
- [Spatial Semantic Segmentation of Sound Scenes](https://dcase.community/challenge2025/task-spatial-semantic-segmentation-of-sound-scenes): Project's inspiration from the DCASE 2025 Task 4 Challenge
- [DCASE2025Task4 Development Dataset](https://zenodo.org/records/15117227)
- [DCASE2025Task4 Evaluation Dataset](https://zenodo.org/records/15553984)
- [DCASE2025Task4 Ground Truth Dataset](https://zenodo.org/records/16736216)

## 🛠 Technologies Used
- Python, NumPy, Pandas, Tensoflow/Keras, Librosa
- EuroHPC Resources
- Overleaf for the syntax of the paper

## 📌 Author
**Georgios Moirotsos**

University of Thessaly, Department of Electrical and Computer Engineering (ECE)

Supervisor: Prof. Gerasimos Potamianos

September 2026
