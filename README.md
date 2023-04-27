This repository contains all important files to reproduce the results of the master thesis "A Survey on Automatic Text Simplification for German".

# Content

## Notebooks
Here you can find all the notebooks used to prepare the data, train the models and then evaluate them.
The notebooks that need to access the corpus data download it as a separate repository.
If you run the notebooks on google colab you have to adjust the path of the prefixes.

The numbering of the notebooks should help you to execute them in the right order. 
For example 02_finetune_decoder or 03_pretraining should always be executed before 04_finetuning, because 04 needs the trained models of 02 or 03.
However, you can also completely skip experiments 02 and 03 for the data augmentation experiments.

The easiest way to implement MBart models with custom decoder is to use the EncoderDecoderModel from huggingface:
```python
#load pre-trained models into one EncoderDecoder
mbart = MBartModel.from_pretrained(mbart_path)
decoder = AutoModelForCausalLM.from_pretrained(decoder_path)
model = EncoderDecoderModel(encoder=mbart.encoder,decoder=decoder)

#push model
model.push_to_hub("custom-mbart")

#load fine-tuned model again
model = EncoderDecoderModel.from_pretrained("custom-mbart") #encoder weights are newly initialized as MBartEncoder is no official encoder architecture
mbart =  MBartModel.from_pretrained("custom-mbart", config=model.config.encoder) #wrap the custom-mbart weights into a full MBartModel
model.encoder = mbart.encoder #set the EncoderDecoders encoder to the fine-tuned version
```

## Results
This folder contains .txt files with the translations of the respective models.
The first file hierarchy divides into 20min and Kurier experiments.  
The second hierarchy divides into experiments done on the original mBART and such done on the custom mBART (with custom decoder).  
The 3th stage finally represents the different main groups of experiments.
- data-augmentation
- finetuned mbart decoders
- pre-training of cross-attentions
- ablation studies

Each of these folders leave folder contains 5 or 10 .txt files with translation candidates.  
Additionally there are 4 .txt files for each experiment.
- (...)-best.txt : containing the best translation candidate according to the reference translation (BLEU)
- (...)-median.txt : containing the median ranked translation candidate according to the reference translation (BLEU)
- (...)-worst.txt : containing the worst translation candidate according to the reference translation (BLEU)
- (...)-most-similar.txt : containing the translation candidate which is most similar to all other candidates (ROUGE)

Evaluating the systems with 06_evaluate_translations I used the (...)-most-similar.txt files as they are independent of the reference translations. The refs.txt and sources.txt files can be found in the corresponding 20min or Kurier folder.

Since the test data set is filtered for samples that do not exceed the maximum number of mBART tokens of 1024 tokens, the composition of the test set depends on the tokenizer (and, in the case of the robustness test, on the augmentation method).For each prediction, you can find the corresponding refs.txt (references) and source.txt (input sequences) by going up the folder hierarchy from your prediction file until you come across res.txt and source.txt for the first time.


It is recommended to use 06_evaluate_translations.ipynb in google colab, as you can simply drag and drop the needed .txt files in the sidebar.

# Questions?
Ask: joshua.oehms@tum.de

