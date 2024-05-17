# BasicBERT: Metaphor Detection via Explicit Basic Meanings Modelling

This repository contains the implementation for the final project of the course COMPSCI 685 - Advanced NLP. This repository is the extension of the work from the ACL 2023 paper "Metaphor Detection via Explicit Basic Meanings Modelling" (https://arxiv.org/pdf/2305.17268.pdf). 

This paper uses two concepts of MIP (Metaphor Identification Process) and SPV (Selectional preference violation) for identifying metaphor 

We propose extension of this project by in two direction : 

1. Utlize GPT 4 for improving the MIP representation 

2. Design a new way to represent SPV. 


## Repository Structure
The repository is structured as follows:

DataCollection/: Contains the script used for extending data and also contains dataset created using GPT3.5.

utils/: Utility functions and scripts for data processing and model evaluation.

results/: Directory for storing evaluation results and metrics.

To run the experiments, please make sure the data needed for running, which is available at 


## Run

For running extensions of our proposed approach, it is evaluated against two more experiments : 

All the below experiments involves changing main_config.cfg : 

1. Running a Vanilla RoBERTA on the dataset :

2. To run Zero-shot with Mistral V0.2 :

3. To run with modified MIP :

4. To run with Modified SPV :



## License

This project is licensed under the MIT License.

## Contact

For questions about our paper or code, please open an issue.

-----
