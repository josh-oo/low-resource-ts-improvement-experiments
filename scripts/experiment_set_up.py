# -*- coding: utf-8 -*-

import sys

import torch
from transformers import set_seed
from transformers import EncoderDecoderModel, AutoModelForCausalLM, LongformerModel
from transformers import MBartModel, MBartTokenizer, MBartConfig
from transformers import BertModel, BertTokenizer, BertConfig
from transformers import GPT2TokenizerFast, GPT2Tokenizer

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import unicodedata

from abc import ABC, abstractmethod
from typing import Iterable
import torch.utils.data
import PIL.Image

import os
import random

from simctg.lossfunction import SimCTGLoss
from transformers import Seq2SeqTrainer

from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

import evaluate

PREFIX = "/../../leichte-sprache-corpus/aligned/20min/"
PREFIX_AUGMENTED = "/../../leichte-sprache-corpus/aligned/20min/augmented/"
PRETRAINED_MODEL = "/../../pretrained_weights.bin"
MAX_INPUT_LENGTH = 1024

EXPERIMENT_NAME = sys.argv[1]
SEED = int(sys.argv[2])
EXPERIMENT_SET = sys.argv[3] #"inputs_back_google.csv"
EXPERIMENT_SET_PROPORTION = float(sys.argv[4]) # 0.5

"""#Prepare Model"""

device = "cuda:0" if torch.cuda.is_available() else "cpu"

encoder_path = "facebook/mbart-large-cc25"
decoder_path = "josh-oo/german-gpt2-easy"

#prepare output_tokenizer

def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
  outputs = token_ids_0 + [self.eos_token_id]
  return outputs

GPT2Tokenizer.build_inputs_with_special_tokens = build_inputs_with_special_tokens

output_tokenizer_fast = GPT2TokenizerFast.from_pretrained(decoder_path)
output_tokenizer_fast.save_pretrained("fast_tokenizer")
output_tokenizer = GPT2Tokenizer.from_pretrained("fast_tokenizer")

output_tokenizer.pad_token_id = 1
output_tokenizer.bos_token_id = 0
output_tokenizer.eos_token_id = 2

EncoderConfig = MBartConfig
EncoderModel = MBartModel
EncoderTokenizer = MBartTokenizer

input_tokenizer = EncoderTokenizer.from_pretrained(encoder_path)
if hasattr(input_tokenizer, "src_lang"):
  input_tokenizer.src_lang = "de_DE"

def get_model():
  encoder_config = EncoderConfig.from_pretrained(encoder_path)

  encoder = EncoderModel.from_pretrained(encoder_path, config=encoder_config)
  encoder = encoder.encoder

  decoder = AutoModelForCausalLM.from_pretrained(decoder_path)
  model = EncoderDecoderModel(encoder=encoder,decoder=decoder)

  # set decoding params
  model.config.decoder_start_token_id = output_tokenizer.bos_token_id
  model.config.eos_token_id = output_tokenizer.eos_token_id
  model.config.pad_token_id = 1
  model.config.max_length = 1024

  #freeze all
  for param in model.parameters():
      param.requires_grad = False

  #make cross attention trainable
  for module in model.decoder.transformer.h:
    for param in module.crossattention.parameters():
      param.requires_grad = True
    for param in module.ln_cross_attn.parameters():
      param.requires_grad = True

  if hasattr(model,'enc_to_dec_proj'):
    model.enc_to_dec_proj.requires_grad = True

  #unfreeze batchnorms
  for module in model.modules():
    if isinstance(module, torch.nn.LayerNorm):
      for param in module.parameters():
        param.requires_grad = True

  #update cross attention
  all_states = model.state_dict()
  update_states = torch.load(PRETRAINED_MODEL)
  all_states.update(update_states)
  model.load_state_dict(all_states)

  return model

"""#Prepare Data"""

class AbstractDataset(torch.utils.data.Dataset, ABC):

    def __init__(self, target_text, source_text=None, input_tokenizer=None, output_tokenizer=None, stride_length=0, max_input_size=512, attentions=None):
        """
        text_dataframe: pandas dataframe with columns topic, phrase
        """
        self.input_tokenizer = input_tokenizer
        self.output_tokenizer = output_tokenizer

        if source_text is not None:
          self.encodings = self.input_tokenizer(
            source_text,
          )
          self.target_text = target_text
        else:
          self.encodings = self.input_tokenizer(
            target_text,
            truncation=True,
            max_length=max_input_size,
            return_overflowing_tokens=True,
            stride = stride_length,
          )

          self.target_text = []
          for input_id in self.encodings['input_ids']:
            
            original_text = self.input_tokenizer.decode(input_id, skip_special_tokens=True)
            self.target_text.append(original_text)


        if self.output_tokenizer is not None:
          output_tokens = self.output_tokenizer(self.target_text,
                                                max_length=1024,
                                                truncation=True,
                                                )

          self.encodings_out = output_tokens['input_ids']
          self.decoder_attention_mask = output_tokens['attention_mask']

        self.attentions = attentions
        self.ids = list(range(0, len(self.encodings['input_ids'])))

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.encodings_out:
          #do not set "decoder_input_ids" and "decoder_attention_mask" as it is set automatically 
          #in this transformers version
          #setting it yourself can lead to strange training results

          item["labels"] = torch.tensor(self.encodings_out[idx])
          item["labels"][item["labels"] == self.output_tokenizer.pad_token_id] = -100  
        else:
          item["labels"] = torch.tensor(self.encodings['input_ids'][idx].copy())
        
        if self.attentions is not None:
          try:
            id = self.ids[idx]
            image = PIL.Image.open(f'{self.attentions}/cross_attn_{id}.tif')
            item["attentions"] = torch.tensor(np.array(image)).to(torch.float16)
          except:
            pass
        return item

    def __len__(self) -> int:
        """
        Returns number of samples in data set

        :return: int - number of samples in data set
        """
        return len(self.ids)


class CombinedDataset(torch.utils.data.ConcatDataset):

    def __init__(self, datasets: Iterable[AbstractDataset]):
        super(CombinedDataset, self).__init__(datasets)

    def get_names(self) -> Iterable[str]:
        """
        Returns a list with the names of all data set that are contained in this combined data set

        :return: list - names of data sets in the data set collection
        """

        return [ds.get_name() for ds in self.datasets]

    def get_summary(self) -> str:
        total_items = 0
        individual_items = {}
        for dataset in self.datasets:
          individual_items[dataset.get_name()] = len(dataset)
          total_items += len(dataset)

        for key in individual_items.keys():
          individual_items[key] = "{:.2f}%".format((individual_items[key]/total_items)*100)
        
        return f"Dataset contains {total_items} items {individual_items}"
        

class ParallelData(AbstractDataset):
    def __init__(self, name, simple_text, normal_text, input_tokenizer, output_tokenizer=None, max_input_size=512, attentions=None):
      self.name = name
      super().__init__(target_text=simple_text,source_text=normal_text, input_tokenizer=input_tokenizer, output_tokenizer=output_tokenizer,max_input_size=max_input_size,attentions=attentions)

    @staticmethod
    def load_from_dataframe(name, csv_file, start_ind, end_ind, input_tokenizer, output_tokenizer=None,max_input_size=512,attentions=None, add_control_tokens=False):
      text_dataframe = pd.read_csv(csv_file)
      if 'phrase_number' in text_dataframe.columns:
        text_dataframe = text_dataframe.replace(np.nan, '', regex=True)
        text_dataframe = text_dataframe.sort_values(['phrase_number']).groupby(['article_id']).agg({'normal_phrase': '\n'.join,'simple_phrase': '\n'.join})
        text_dataframe = text_dataframe.replace(r'\r+|\n+|\t+',' ', regex=True)
      
      if add_control_tokens == True:
        def custom_round(x, base=5):
          return min(round(base * round(float(x)/base), 1), 2.0)

        base = 0.1
        lev_sim = text_dataframe['lev_sim'].apply(lambda x: custom_round(x, base=base)).astype(str)
        word_rank = text_dataframe['rank'].apply(lambda x: custom_round(x, base=base)).astype(str)
        dep = text_dataframe['dep'].apply(lambda x: custom_round(x, base=base)).astype(str)
        nbchars = text_dataframe['nbchars'].apply(lambda x: custom_round(x, base=base)).astype(str)

        tokens = "<NbChars_" + nbchars + "><LevSim_" + lev_sim + "><WordRank_" + dep + "><DepTreeDepth_" + word_rank + ">"

        text_dataframe['normal_phrase'] =  tokens + text_dataframe['normal_phrase']
      if isinstance(add_control_tokens, str):
        text_dataframe['normal_phrase'] =  add_control_tokens + text_dataframe['normal_phrase']

      if (end_ind < 0):
        text_dataframe = text_dataframe[start_ind:][:]
      else:
        text_dataframe = text_dataframe[start_ind:end_ind][:]

      simple_text = [ unicodedata.normalize("NFC",s.strip()) for s in list(text_dataframe['simple_phrase'].values)]
      normal_text = [ unicodedata.normalize("NFC",s.strip()) for s in list(text_dataframe['normal_phrase'].values)]
      return ParallelData(name, simple_text, normal_text, input_tokenizer, output_tokenizer=output_tokenizer,max_input_size=max_input_size,attentions=attentions)

    @staticmethod
    def load_from_parallel_files(name, src_file, tgt_file, start_ind, end_ind, input_tokenizer, output_tokenizer=None,max_input_size=512):
      src_file = open(src_file)
      tgt_file = open(tgt_file)

      simple_text = []
      normal_text = []
      parallel = zip(src_file, tgt_file)

      for text_pair in parallel:
        normal_text.append(unicodedata.normalize("NFC",text_pair[0].strip()))
        simple_text.append(unicodedata.normalize("NFC",text_pair[1].strip()))
      
      return ParallelData(name, simple_text, normal_text, input_tokenizer, output_tokenizer=output_tokenizer,max_input_size=max_input_size)

    def get_name(self) -> str:
      return self.name

    def get_columns(self) -> Iterable[str]:
      return self.texts.columns

class MonolingualData(AbstractDataset):
    def __init__(self, name, csv_file, stride_length,input_tokenizer, output_tokenizer=None):
        phrases = pd.read_csv(csv_file).fillna('text')
        self.texts = phrases.sort_values(['phrase_number']).groupby(['topic'])['phrase'].apply(' '.join).reset_index()
        simple_text = [ unicodedata.normalize("NFC",s) for s in list(self.texts['phrase'].values)]

        self.name = name
        super().__init__(target_text=simple_text, input_tokenizer=input_tokenizer, output_tokenizer=output_tokenizer)

    def get_name(self) -> str:
      return self.name

    def get_columns(self) -> Iterable[str]:
      return self.texts.columns

def filter_dataset(dataset, filter_size_max=1024, control_tokens=None):
  if control_tokens is not None:
    filter_size_max += len(input_tokenizer(control_tokens)['input_ids']) - 2
  number_of_tokens = []
  pick = []
  for i in range(0, len(dataset)):
    current_size = dataset[i]['input_ids'].shape[0]
    number_of_tokens.append(current_size)
    if current_size < filter_size_max:
      pick.append(i)

  return torch.utils.data.Subset(dataset, pick)

def prepare_data(max_input_length, experiment_set, experiment_proportion=1.0):
  train_set = ParallelData.load_from_dataframe("20min_aligned_train", PREFIX + "20min_aligned_train.csv", 0, -1, input_tokenizer,output_tokenizer,max_input_size=max_input_length)
  val_set = ParallelData.load_from_dataframe("20min_aligned_val", PREFIX + "20min_aligned_dev.csv", 0, -1, input_tokenizer,output_tokenizer,max_input_size=max_input_length)
  test_set = ParallelData.load_from_dataframe("20min_aligned_test", PREFIX + "20min_aligned_test.csv", 0, -1, input_tokenizer,output_tokenizer,max_input_size=max_input_length)

  augmented_set = ParallelData.load_from_dataframe("20min_aligned_augmented", PREFIX_AUGMENTED + experiment_set, 0, -1, input_tokenizer,output_tokenizer,max_input_size=max_input_length)

  train_set = filter_dataset(train_set, filter_size_max=max_input_length)
  val_set   = filter_dataset(val_set, filter_size_max=max_input_length)
  test_set  = filter_dataset(test_set, filter_size_max=max_input_length)

  augmented_set = filter_dataset(augmented_set, filter_size_max=max_input_length)

  #take proportion
  indices = random.sample(range(0, len(augmented_set)), int(len(augmented_set) * experiment_proportion))
  augmented_set = torch.utils.data.Subset(augmented_set, indices)

  return train_set, val_set, test_set, augmented_set

"""#Trainingstuff"""

class CustomCollator(DataCollatorForSeq2Seq):
  def __call__(self, features, return_tensors=None):

    attentions = [feature["attentions"] for feature in features] if "attentions" in features[0].keys() else None
    
    for feature in features:
      feature['input_ids'] = feature['input_ids'][:self.max_length]
      feature['attention_mask'] = feature['attention_mask'][:self.max_length]

    if attentions is not None:
      #align width
      max_width = max(len(a[0,:]) for a in attentions)
      if self.pad_to_multiple_of is not None:
          max_width = (
              (max_width + self.pad_to_multiple_of - 1)
              // self.pad_to_multiple_of
              * self.pad_to_multiple_of
          )

      padding_side = self.tokenizer.padding_side
      for feature in features:
          remainder = [0] *  (max_width - len(feature["attentions"][0,:]))
          remainder = [remainder] * len(feature["attentions"][:,0])
          if padding_side == "right":
              feature["attentions"] = np.concatenate([feature["attentions"], remainder], axis=1).astype(np.float32)
          else:
              feature["attentions"] = np.concatenate([remainder, feature["attentions"]], axis=1).astype(np.float32)

      #align height
      max_height = max(len(a[:,0]) for a in attentions)
      if self.pad_to_multiple_of is not None:
          max_height = (
              (max_height + self.pad_to_multiple_of - 1)
              // self.pad_to_multiple_of
              * self.pad_to_multiple_of
          )

      padding_side = self.tokenizer.padding_side
      for feature in features:
          remainder = [0] * len(feature["attentions"][0,:])
          remainder = [remainder] * (max_height - len(feature["attentions"][:,0]))
          if padding_side == "right" and len(remainder) > 0:
              feature["attentions"] = np.concatenate([feature["attentions"], remainder], axis=0).astype(np.float32)
          elif len(remainder) > 0:
              feature["attentions"] = np.concatenate([remainder, feature["attentions"]], axis=0).astype(np.float32)

    return super().__call__(features, return_tensors)

#adapted from https://github.com/yxuansu/SimCTG

margin = 0.5
vocab_size = len(output_tokenizer)
simctgloss = SimCTGLoss(margin=margin, vocab_size=vocab_size, pad_token_id=output_tokenizer.pad_token_id)

class ContrastiveTrainer(Seq2SeqTrainer):
    def __init__(self, attention_alpha=0.5, attention_temperature=1/16, **kwargs):
      self.attention_alpha = attention_alpha
      self.attention_temperature = attention_temperature

      self.kl_loss = torch.nn.KLDivLoss(reduction="none")
      super().__init__(**kwargs)

    def evaluate(self,eval_dataset = None,ignore_keys = None,metric_key_prefix = "eval",**gen_kwargs):
        return super().evaluate(eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix, num_beams=1, do_sample=False)#, top_k=3, penalty_alpha=0.6)

    def predict(self,test_dataset,ignore_keys = None,metric_key_prefix = "test",**gen_kwargs):
      return super().predict(test_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix, num_beams=3, do_sample=False)

    def shift_tokens_right(self, input_ids, pad_token_id, decoder_start_token_id):
        """
        Shift input ids one token to the right.
        """
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        if decoder_start_token_id is None:
            raise ValueError("Make sure to set the decoder_start_token_id attribute of the model's configuration.")
        shifted_input_ids[:, 0] = decoder_start_token_id

        if pad_token_id is None:
            raise ValueError("Make sure to set the pad_token_id attribute of the model's configuration.")
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids

    def compute_loss(self, model, inputs, return_outputs=False):

        labels = inputs.get('labels')
        input_ids = inputs.get('input_ids')
        attention_mask = inputs.get('attention_mask')

        guide_attention = inputs.pop('attentions', None)

        decoder_input_ids = self.shift_tokens_right(labels,output_tokenizer.pad_token_id,output_tokenizer.bos_token_id)
        decoder_attention_mask = labels != -100

        # forward computation
        bsz, seqlen = labels.size()
        outputs = model(**inputs, output_hidden_states=True, output_attentions=True)
        logits = outputs.logits

        regular_loss = outputs.loss
        if self.label_smoother is not None:
          regular_loss = self.label_smoother(outputs, labels, shift_labels=False)

        attn_loss = torch.tensor(0.0)
        if guide_attention is not None:
          #attention loss
          log_summation = outputs['cross_attentions'][-1]

          attn_mask = torch.bmm(decoder_attention_mask.float().unsqueeze(2), attention_mask.float().unsqueeze(1))
          
          epsilon = 1e-6

          log_summation = torch.add(log_summation, epsilon)
          log_summation = torch.mul(torch.sum(log_summation.log(), dim=1), self.attention_temperature)
          
          #log_summation = log_summation + softmax_mask # set masked pixels to -inf to avoid softmax influenz
          log_summation = log_summation.masked_fill((1 - attn_mask).bool(), float('-inf'))
          log_summation = torch.nn.functional.log_softmax(log_summation, dim=-1)
          log_summation = torch.nan_to_num(log_summation, nan=0.0, neginf=0.0)

          attn_loss = self.kl_loss(log_summation, guide_attention)
          attn_loss = (attn_loss * attn_mask).sum()
          attn_loss = attn_loss / guide_attention.size(0) # to have batch_mean

        assert logits.size() == torch.Size([bsz, seqlen, model.decoder.config.vocab_size])
        last_hidden_states = outputs.decoder_hidden_states[-1]

        # compute cl loss
        norm_rep = last_hidden_states / last_hidden_states.norm(dim=2, keepdim=True)
        cosine_scores = torch.matmul(norm_rep, norm_rep.transpose(1,2)) 
        assert cosine_scores.size() == torch.Size([bsz, seqlen, seqlen])
        cl_loss = simctgloss.contrastive_loss(cosine_scores, decoder_input_ids)

        simctg_loss = (regular_loss + cl_loss) * (1 - self.attention_alpha) + attn_loss * self.attention_alpha
        return (simctg_loss, logits) if return_outputs else simctg_loss

class TestTrainer(Seq2SeqTrainer):
    def __init__(self, gen_kwargs, **kwargs):
      self.gen_kwargs = gen_kwargs
      super().__init__(**kwargs)
    def predict(self,test_dataset,ignore_keys = None,metric_key_prefix = "test",**gen_kwargs):
      return super().predict(test_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix, **self.gen_kwargs)


def compute_translation_metrics(input_tokenizer, output_tokenizer, pred, control_tokens=False):

    input_ids = pred.inputs
    label_ids = pred.label_ids
    pred_ids = pred.predictions

    input_ids[input_ids == -100] = input_tokenizer.pad_token_id
    label_ids[label_ids == -100] = output_tokenizer.pad_token_id
    pred_ids[pred_ids == -100] = output_tokenizer.pad_token_id

    input_str_list = input_tokenizer.batch_decode(input_ids, skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=False)
    pred_str_list = output_tokenizer.batch_decode(pred_ids, skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=False)
    label_str_list = output_tokenizer.batch_decode(label_ids, skip_special_tokens=True,
                                                   clean_up_tokenization_spaces=False)
    
    if control_tokens == True:
      for i in range(0, len(input_str_list)):
        input_str_list[i] = input_str_list[i].split(' ', 1)[1]

    label_str_list = [[label] for label in label_str_list]

    sari = evaluate.load("sari")
    bleu = evaluate.load("bleu")

    sari_score = sari.compute(sources=input_str_list, predictions=pred_str_list, references=label_str_list)
    bleu_score = bleu.compute(predictions=pred_str_list, references=label_str_list)

    translation_result = {
        'sari':sari_score['sari'],
        'bleu':bleu_score['bleu']*100
    }

    return {key: sum(value) / len(value) if isinstance(value, Iterable) else value for (key, value) in
            translation_result.items()}

compute_metrics = lambda pred: compute_translation_metrics(input_tokenizer, output_tokenizer, pred)

"""#Run Experiment"""

if not os.path.exists(EXPERIMENT_NAME):
  os.mkdir(EXPERIMENT_NAME)

if not os.path.exists(os.path.join(EXPERIMENT_NAME, str(SEED))):
  os.mkdir(os.path.join(EXPERIMENT_NAME, str(SEED)))

steps_to_train = 20

set_seed(SEED) # no direct effect on text generation

model = get_model()
train_set, val_set, test_set, augmented_set = prepare_data(MAX_INPUT_LENGTH, EXPERIMENT_SET, EXPERIMENT_SET_PROPORTION)

training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    include_inputs_for_metrics=True,
    generation_max_length=10,
    max_steps=steps_to_train,
    output_dir="/results",
    evaluation_strategy="steps",
    save_strategy='no',
    learning_rate=1e-3, 
    weight_decay=0.01, 
    warmup_steps=100,
    per_device_eval_batch_size=4, 
    per_device_train_batch_size=2, 
    gradient_accumulation_steps=8,
    fp16=True,
    logging_steps=500,
    group_by_length=True,
    seed=SEED,
    data_seed=SEED,
    remove_unused_columns=False,
)

all_training_sets = CombinedDataset([train_set, augmented_set])

data_collator = CustomCollator(tokenizer=input_tokenizer, model=model, pad_to_multiple_of=8, max_length=MAX_INPUT_LENGTH)

trainer = ContrastiveTrainer(
    attention_alpha=0,
    attention_temperature=1/24,
    model=model,
    args=training_args,
    compute_metrics = compute_metrics,
    train_dataset=all_training_sets,
    eval_dataset=val_set,
    data_collator=data_collator,
)
trainer.train()
preds = trainer.predict(test_dataset=test_set)

with open(os.path.join(EXPERIMENT_NAME,str(SEED),'logs.txt'), 'w') as fp:
    for item in trainer.state.log_history:
      fp.write(str(item) + '\n')

with open(os.path.join(EXPERIMENT_NAME,str(SEED),'beam_3_log.txt'), 'w') as fp:
    fp.write(str(preds.metrics))

outputs = output_tokenizer.batch_decode(preds.predictions, clean_up_tokenization_spaces=True, skip_special_tokens=True)
for i in range(0,len(outputs)):
  outputs[i] = outputs[i].replace("\n", " ")
with open(os.path.join(EXPERIMENT_NAME,str(SEED),'results.txt'), 'w') as fp:
    fp.write("\n".join(outputs))

def run_with(settings):
  tester = TestTrainer(
      gen_kwargs=settings,
      model=model,
      args=training_args,
      compute_metrics = compute_metrics,
      train_dataset=all_training_sets,
      eval_dataset=val_set,
      data_collator=data_collator,
  )
  return tester.predict(test_dataset=test_set)

preds_1 = run_with({'num_beams':1, 'do_sample':False})
with open(os.path.join(EXPERIMENT_NAME,str(SEED),'beam_1_log.txt'), 'w') as fp:
    fp.write(str(preds_1.metrics))

preds_2 = run_with({'num_beams':2, 'do_sample':False})
with open(os.path.join(EXPERIMENT_NAME,str(SEED),'beam_2_log.txt'), 'w') as fp:
    fp.write(str(preds_2.metrics))

preds_4 = run_with({'num_beams':4, 'do_sample':False})
with open(os.path.join(EXPERIMENT_NAME,str(SEED),'beam_4_log.txt'), 'w') as fp:
    fp.write(str(preds_4.metrics))
