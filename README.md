# Separation-Classification-and-Localization-of-Acoustic-Events-from-Multi-channel-Audio-Signals

This repository contains the Python code developed for my Diploma Thesis project, as well as the Thesis paper and presentation. In this Thesis we analyze complex acoustic scenes by developing a three-stage system for separating, classifying, and localizing overlapping acoustic events in multichannel audio mixture recordings.

## Project Description
We will be using multichannel audio mixture recordings with a duration of 10 seconds in a .wav format, provided by the DCASE 2025 Task 4 Dataset. Each mixture is captured by four microphone channels, containing between 1 to 3 different acoustic events which occur simultaneously. Additional noise and interference sounds may be included into the mixtures. We focus mainly on the isolation of the individual sound sources on the mixture. This procedure takes place without any knowledge of the mixing process or the existing sources, which is called Blind Source Separation (BSS). Then, each isolated target sound event will be assigned a predicted label of an acoustic event class. Lastly,  we estimate the direction of arrival of each separated source signal, by exploiting the spatial information captured by the microphone array. 

## Features
This project can be divided into three parts:
- **Blind Source Separation**: The first task aims to recover a set of audio source signals from the observation of the multichannel mixture signal. The separation stage considers Indevendent Vector Analysis (IVA). The main assumption of the IVA algorithm is the statitiscal independence of the source signals and their spatial diversity across the microphone channels.
- **Acoustic Event Classification**: The separated signals will be fed into a Convolutional Neural Network (CNN) architecture in order to predict their class label. There is a total of 18 acoustic event classes which differ in duration, intensity, spectral content, and temporal structure, making the classification problem extremely challenging. The CNN model has been trained on predefined clean target source signals from each class, provided by the DCASE 2025 Task 4 Development Dataset.
- **Acoustic Source Localization**: We estimate the spatial direction of each separated source signal, by exploiting the spatial information captured by the microphone array. The estimation of the direction of arrival of each source is computed in terms of azimuth and elevation coordinates. Localization is not treated as an independent processing step. In contrast, is is recovered from the separated outputs by exploiting spatial information obtained during the preceding source separation stage.

## Repository Structure

| Path | Contents |
| --- | --- |
| [`separation`](separation/) | Python backend with Flask routes, the MySQL connection, and SQL queries for the application features. |
| [`classification`](classification/) | SQL dump containing the relational schema and project data. |
| [`localization`](localization/) | Frontend template with HTML, CSS, and JavaScript. |
| [`evaluation`](evaluation/) | Frontend template with HTML, CSS, and JavaScript. |
| [`Moirotsos_georgios.pdf`](Moirotsos_georgios.pdf) | Project report with installation instructions, schema diagrams, screenshots, and a code walkthrough. |
| [`Thesis_presentation.pptx`](Thesis_presentation.pptx) | Project presentation. |
| [`complete_pipeline.png`](complete_pipeline.png) | Project complete pipeline. |

## Thesis paper
Anyone looking for a more detailed and deeper understanding of the project as well as the final results and conclusions of my work can take a look at:
https://www.e-ce.uth.gr/wp-content/uploads/formidable/59/Moirotsos_georgios.pdf
