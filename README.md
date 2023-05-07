# TranscriptsSuicideML

Due to legal reasons we could not share the training data with the public. If you want to train your own models, make sure to add a **/data/new** directory that contains raw text files. We provide a sample text file for reference. You can upload your own data and run it through the train or predict pipeline to create your own results.  

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

The trained models can be found and loaded via their variable codes in dockerhub:

 - https://huggingface.co/HubertBaginski/BERT_AU01_01
 - https://huggingface.co/HubertBaginski/BERT_PS01
 - https://huggingface.co/HubertBaginski/BERT_PR01_01
 - https://huggingface.co/HubertBaginski/BERT_PO01_01
 - https://huggingface.co/HubertBaginski/BERT_MF02_01
 - https://huggingface.co/HubertBaginski/BERT_MF02_03
 - https://huggingface.co/HubertBaginski/BERT_MF02_12
 - https://huggingface.co/HubertBaginski/BERT_MF01
 - https://huggingface.co/HubertBaginski/BERT_ID05_01
 - https://huggingface.co/HubertBaginski/CS02_01


- Todo:
  - set dockerhub image to public
  - set docker repo to public
  - add descirption info to the huzggignface models (update with link to repo and paper)
