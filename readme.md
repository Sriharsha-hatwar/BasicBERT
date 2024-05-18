# BasicBERT: Metaphor Detection via Explicit Basic Meanings Modelling

This repository contains the implementation for the final project of the course COMPSCI 685 - Advanced NLP. This repository is the extension of the work from the ACL 2023 paper "Metaphor Detection via Explicit Basic Meanings Modelling" (https://arxiv.org/pdf/2305.17268.pdf). 

This paper uses two concepts of MIP (Metaphor Identification Process) and SPV (Selectional preference violation) for identifying metaphor in the BasicBERT model which we use as our starting point of experiments.

We propose extension of this project by in two direction : 

1. Adding new basic sentences from Merriam-Webster dictionary API and GPT4 to improving the MIP representation 

2. Design a new way to represent SPV. 


## Repository Structure
The repository is structured as follows:

DataCollection/: Contains the scripts for obtaining literal sentences from Merriam-Webster API and formatting sentnces into VUA20 dataset format. Also contains generated sentences from GPT-4. 

utils/: Utility functions and scripts for data processing and model evaluation.

To run the experiments, please make sure the data needed for running, which is available at https://huggingface.co/datasets/liyucheng/vua20/tree/main


## Run

For running extensions of our proposed approach, it is evaluated against two more experiments, hence 5 : 

All the below experiments involves changing main_config.cfg : 

1. Running a Vanilla RoBERTA on the dataset :
   ```
   python main.py --model_type MELBERT --bert_model roberta-base --learning_rate 5e-5 --seed 41
   ```

2. To run Zero-shot with Mistral V0.2 :
   This involves running the file : zero-shot.py
    ```
    python zero-shot.py 
    ```
    If you want to change the location of the test dataset, please change it in line 109. 

3. To run BasicBERT with improved MIP :

   This experiment involves changing the training dataset where we can use this : https://drive.google.com/file/d/1Io2PQlL6k1IN1vFOARuqacs4NqP_3tgY/view?usp=sharing
   and run the file :
   ```
   python main.py
   ```
   
4. To run modified SPV :
   
   Change the main_config.cfg where the model_type is changed to `SPV_MODIFIED` and task_name is `vuaextended` which contains the new dataset.
   later, exectute :
   ```
   python main.py
   ```
   This experiment also includes using the training dataset : https://drive.google.com/file/d/1J3wG9bn5bJOphsPpmR4X9BS9eJzGaEm3/view?usp=sharing
   please replace this in data/VUA20 and execute.
   To run the inference, please make sure `run_train=False` and `run_infer=True` with the model at : https://drive.google.com/file/d/1kDvi6ldVi-GWWd1YFlipLoUVTBvordkS/view?usp=drive_link
   with `use_finetune_path` pointing to the saved model.


## Saved models and Saved dataset : 

Here is the location for the saved dataset and finetuned models : https://drive.google.com/drive/folders/1RkXznO0N2a_Adf-aipSHXFSc3Wjs6BDn?usp=sharing



## License

This project is licensed under the MIT License.

## Contact

For questions about our paper or code, please open an issue.

-----
