# Potential Neutralizing Antibodies Discovered for Novel Corona Virus Using Machine Learning
This github repository is for our paper "Potential Neutralizing Antibodies Discovered for Novel Corona Virus Using Machine Learning" published in Scientific Reports. To read the paper please visit the following link: https://www.nature.com/articles/s41598-021-84637-4

# Dataset - VirusNet
We have collected the dataset from LANL - CATNAP database and the RCSB PDB server. The VirusNet database is a collection of the FASTA sequence of the antibody and the corresponding antigen sequence. The process of data collection is described in the paper.

VirusNet.csv is the dataset that we used in the paper. We have also added some additional data to the existing VirusNet data, this additional data is present in VirusNet_additional.csv. 
Please cite CATNAP (https://www.hiv.lanl.gov/components/sequence/HIV/neutralization/) if you use the VirusNet data and the instructions for citing PDB resources can be found at this link:https://www.rcsb.org/pages/policies

# Methods
We used graph featurization of the FASTA sequence by converting the sequence into its corresponding molecule and then using atom features to create a representation. After creating the representation, we ran some of the standard machine learning models to determine the antibodies that can neutralize SARS-CoV-2. These antibodies were later validated through molecular dynamics simulations. More details about the methods can be found in the paper. 

# Running the code 
The packages that we used in the code numpy, scikit-learn, XGBoost, matplotlib, rdkit. The instructions for installing them can be found on the webpages for the respective packages. To run the code on your custom data just replace the filename at the appropriate commented place in the code. After installing the packages

# Authors 
The work was done by Rishikesh Magar and Prakarsh Yadav under the supervision of Amir Barati Farimani
