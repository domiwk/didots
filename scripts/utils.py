import pandas as pd
import json
import stanza
import numpy as np

from stanza.pipeline.processor import ProcessorVariant, register_processor_variant, Processor,register_processor


def save_as_json(preds, test_id_path, out_path_dict):
  data = pd.read_csv(f'{test_id_path}/meta_data_test.txt', sep=';')
  ids= data['ID   '].tolist()
  ids = [x.strip() for x in ids]

  preds_dict = dict()
  for idx, x in enumerate(preds):
    pred_array = [ids,int(x)] 
    preds_dict[ids[idx]] = pred_array
  #print(preds_dict)
    
  with open(out_path_dict, 'w+') as fp:
        json.dump(preds_dict, fp)


def get_constituency(docs):
  cnst_list = []
  for doc in docs:
    for sentence in doc:
      cnst_list.append(sentence.constituency)
  return cnst_list

def preprocess_pipe(texts, nlp):
    preproc_pipe = []
    for doc in nlp.pipe(texts, batch_size=20):
        preproc_pipe.append(get_constituency(doc, nlp))
    return preproc_pipe


@register_processor("doc_constituency")
class DocConstituency(Processor):
    ''' Processor that lowercases all text '''
    _requires = set(['tokenize'])
    _provides = set(['lowercase'])

    def __init__(self, config, pipeline, use_gpu):
        pass

    def _set_up_model(self, *args):
        pass

    def process(self, doc):
        #doc.text = doc.text.lower()
        cnst_list = []
        for sent in doc.sentences:
          cnst_list.append(sent.constituency)

        doc.constituency = cnst_list
        return doc

def compute_doc_constituency(df = None, dataset_path = None):
  if df is None:
    df = pd.read_csv(dataset_path)    

  print(df.head())
  #nlp = English()
  nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency,doc_constituency', use_gpu=True, pos_batch_size=16)


  texts = [stanza.Document([], text=d) for d in df['AdvText']]
  
  if 'AdvText' in df.columns:
    texts = [ ""  if d is np.nan else d for d in df['AdvText']]
  else:
    texts = [ "" if d is np.nan  else d for d in df['Text']]

  in_docs = [stanza.Document([], text=d) for d in texts]

  out_docs = nlp(in_docs)

  docs_const = [d.constituency for d in out_docs]

  df['Constituency'] = docs_const

  return df


def compute_sent_constituency(df = None, dataset_path = None):
  if df is None:
    df = pd.read_csv(dataset_path)    

  print(df.head())
  #nlp = English()
  nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency,doc_constituency', use_gpu=True, pos_batch_size=16)
  
  if 'AdvText' in df.columns:
    texts = [stanza.Document([], text=d) for d in df['AdvText']]
    texts = [ ""  if d is np.nan else d for d in df['AdvText']]
  else:
    texts = [stanza.Document([], text=d) for d in df['Text']]
    texts = [ "" if d is np.nan  else d for d in df['Text']]


  in_docs = [stanza.Document([], text=d) for d in texts]
  out_docs = nlp(in_docs)

  docs_const = [ d.constituency[0] if len(d.constituency) > 0 else "" for d in out_docs ]

  df['GeneratedConstituency'] = docs_const

  return df


def split_into_sentences_and_constituency(df = None, dataset_path = None):
  if df is None:
    df = pd.read_csv(dataset_path)    

  print(df.head())
  #nlp = English()
  nlp = stanza.Pipeline(lang='en', processors='tokenize, pos, constituency, doc_constituency', use_gpu=True, pos_batch_size=16)
  
  if 'AdvText' in df.columns:
    in_docs = [stanza.Document([], text=d.replace('[SEP]','')) for d in df['AdvText']]
  else:
    in_docs = [stanza.Document([], text=d.replace('[SEP]','')) for d in df['Text']]
  out_docs = nlp(in_docs)

  if 'Intent' in df.columns:
    df_intents = [d for d in df['Intent']]
  else:
    df_intents = [d for d in df['Class']]

  sentences = []
  constituencies = []
  ids = []
  intents = []

  for idx, d in enumerate(out_docs):
    sentences_text = [s.text for s in d.sentences ]
    #print(sentences_text)

    sentences.extend(sentences_text)
    constituencies.extend(d.constituency)

    #print(constituencies)
    #print(df_intents[idx])

    ids.extend(len(sentences_text)*[idx])
    intents.extend(len(sentences_text)*[df_intents[idx]])


  print(sentences_text[:10])
  sent_df = pd.DataFrame(zip(sentences,constituencies,intents,ids), columns = ["Text","Constituency","Intent","ID"])

  return sent_df

