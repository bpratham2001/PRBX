# PRBX

The objective of this paper was to explore a small part of the process of training a neural network, that is the data augmentation done during training. By augmenting the data used for testing, we can discover edge case behaviours to help us better understand the decisions made by the model. By augmenting the data used in training, not only can we work with limited data, but also influence the training such that the model learns relevant features to make better decisions for edge and adversarial cases.

Due to all of the top performing models in the Udacity Self-Driving 
Car Challenge being LSTM-based, the objective of this project narrows itself down to the exploration of the effects of data augmentation when training and testing, specifically on an LSTM based steering model. It was found that not only does data augmentation reduce the error from testing on augmented and unaugmented data, but it also significantly reduces zero-shot testing error.

In this folder:
1) The main document is under the name "pb1028_DataAugmentation"
2) The tables and graphs used in the main document are in the file "RMSEresults.xlsx"
3) The diagrams used in the main document are in the file "diagrams.pptx"
4) The personal reflections document on the project is under the name "pb1028_PersonalReflections"

The code (PRBX_komanda), along with the datasets (HMB_*.bag) referenced in the main document
can be found in the Google Drive folder of the link below:
https://drive.google.com/drive/folders/18Ojv3uEk3XI_XF7Qc3lHj6olv49P1Nu9?usp=sharing
