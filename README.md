# Potential Neutralizing Antibodies Discovered for Novel Corona Virus Using Machine Learning
This github repository is for our paper "Potential Neutralizing Antibodies Discovered for Novel Corona Virus Using Machine Learning" published in Scientific Reports. To read the paper please visit the following link: https://www.nature.com/articles/s41598-021-84637-4

# Dataset - VirusNet
We have collected the dataset from LANL - CATNAP database and the RCSB PDB server. The VirusMet database is a collection of the FASTA sequence of the antibody and the corresponding antigen sequence. The process of data collection is described in the paper

# Methods
We used graph featurization of the FASTA sequence by converting the sequence into its corresponding molecule and then using atom features to create a representation. After creating the representation, we ran some of the standard machine learning models to determine the antibodies that can neutralize SARS-CoV-2. These antibodies were later validated through molecular dynamics simulatiosn. More details about the methods can be found in the paper. 

