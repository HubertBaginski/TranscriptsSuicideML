# Code and Models for the paper: "A machine learning approach to detect potentially harmful and protective suicide-related content in broadcast media"

## Data

Because the transcripts of media broadcasts are proprietary data (by Brandwatch) we could not share the training data with the public. If you want to train your own models, make sure to add a **/data/new** directory that contains raw text files. We provide a sample text file for reference. You can upload your own data and run it through the train or predict pipeline to create your own results.

In data/new/ you can find 

- an example transcript in raw text format
- the datafile "V!brant_data_all_161220.csv" (as .csv and .xlsx) containing the coded characteristics for all transcripts
- Detailed information about how all characteristics were coded/labelled, in the file "manual_coding_system_online_questionnaire.pdf". It's in the format of a SoSci Survey online questionnaire, in which all transcripts were coded. 

### Documentation for the datafile "V!brant_data_all_161220.csv"

- First row: Variable codes for each characteristic
- Second row: Explanation of variable code, summary of the label
- Row 3 onward: codes for all characteristics per transcript, many more characteristics than those for which we trained models. One line per transcript. 
- Example characteristic: 
    - Variable code: AU01_01
    - Label summary: Alternatives to suicidal behaviour
    - If you search AU01_01 in the PDF with the coding system, you'll find detailed labelling explanations on page 36 (" The item includes alternatives to suicidal behaviour. ANY alternative
counts. ...") , and the labels for numeric codes in the datafile: 1= No, 2 = Yes, -9 = Not answered.
- Column 1: "Interview number (ongoing)" - Interview is an automatic label by Soscisurvey, it refers to transcripts here. 
- Note on sample size: the sample of transcripts in the .csv file is higher than the number used for model training in the paper, because the datafile also contains online news articles, which were not analysed in the current study.

## Model training

If you want to train and predict the models yourself, you can use the docker image and the docker-compose files. If you have an NVIDIA GPU the training will run on the GPU, please adjust the following parameters in the **docker-compose.yml**:
 - **--batch_size**: set according to memory of GPU, if < 10GB use 2, else 4
 - **--epochs**: maxmimum number of epochs
 - **--variable_code**: the variable code you want to investigate (see https://github.com/HubertBaginski/TranscriptsSuicideML/blob/5ec17fd92612bf8ef35ec3c37635939ff8ffb887/utils/variable_codes.py for reference)
 - **volumes**: mount a local directory that exists on your file system -> /mypath/my_folder:/checkpoint_folder
 - **data**: mount a local directory that contains the training documents -> /mypath/my_data:/data/new


Once you modified the **docker-compose.yml** file, you can run it with `docker-compose -f docker-compose.yml up`

The training will create one weights file for each epoch and print the evaluation results to the console. Manually select the "best" epoch - balanced between training and evaluation performance - and paste the name into the **docker-compose-predict.yml**:
 - **--batch_size**: set according to memory of GPU, if < 10GB use 2, else 4
 - **--weights_file**: name of the "best" weights file you want to use for testing, e.g.: weights-13.hdf5
 - **--epochs**: set to 1, will train 1 epoch, and then load the weights file you specified.
 - **--checkpoint_folder** /checkpoint_folder
 - **--variable_code**: use the same variable code you used in the training
 - **volumes**: update path to match checkpoint folder specified in training -> /mypath/my_folder:/checkpoint_folder
 - **data**: mount a local directory that contains the test documents -> /mypath/my_data:/data/new

You can also set up the environment locally by installing the requirements via `pip3 install -r requirments.txt`. Then you can use the scripts `run_transcripts_bert.py` for training and `predict_transcripts_bert.py` for inference. You can also work with the jupyter notebook that offers additional comments.


## Machine learning models for download

The trained models can be found and loaded via their variable codes in dockerhub:

 - Suicide death (completed suicide): https://huggingface.co/HubertBaginski/bert-suicide-death-completed
 - Celebrity suicide: https://huggingface.co/HubertBaginski/bert-celebrity-suicide
 - Alternatives to suicide: https://huggingface.co/HubertBaginski/bert-alternatives-to-suicide
 - Monocausality: https://huggingface.co/HubertBaginski/bert-monocausality
 - Positive outcome of suicidal crisis: https://huggingface.co/HubertBaginski/bert-positive-outcome-crisis
 - Healing story: https://huggingface.co/HubertBaginski/bert-healing-story
 - Suicidal ideation: https://huggingface.co/HubertBaginski/bert-suicidal-ideation
 - Problem vs. Solution focus: https://huggingface.co/HubertBaginski/bert-problem-or-solution
 - Enhancing myths: https://huggingface.co/HubertBaginski/bert-enhancing-myths
 - Main focus: https://huggingface.co/HubertBaginski/bert-main-focus
