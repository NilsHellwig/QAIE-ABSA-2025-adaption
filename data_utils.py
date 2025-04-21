# -*- coding: utf-8 -*-

# This script contains all data transformation and reading

import random
from torch.utils.data import Dataset
import spacy 

senttag2word = {'POS': 'positive', 'NEG': 'negative', 'NEU': 'neutral'}
senttag2opinion = {'POS': 'great', 'NEG': 'bad', 'NEU': 'ok'}
sentword2opinion = {'positive': 'great', 'negative': 'bad', 'neutral': 'ok'}
sentim = ['great', 'bad', 'ok']
aspect_cate_list = ['location general',
                    'food prices',
                    'food quality',
                    'food general',
                    'ambience general',
                    'service general',
                    'restaurant prices',
                    'drinks prices',
                    'restaurant miscellaneous',
                    'drinks quality',
                    'drinks style_options',
                    'restaurant general',
                    'food style_options']


def read_line_examples_from_file(data_path, silence):
    """
    Read data from file, each line is: sent####labels
    Return List[List[word]], List[Tuple]
    """
    reviews, sents, labels = [], [], []
    with open(data_path, 'r', encoding='UTF-8') as fp:
        words, labels = [], []
        for line in fp:
            line = line.strip()
            if line != '':
                words, tuples = line.split('####')
                reviews.append(words)
                sents.append(words.split())
                labels.append(eval(tuples))
    if silence:
        print(f"Total examples = {len(sents)}")
    # print(sents[:2])
    # print(reviews[:2])
    # print(labels[:2])
    return sents, reviews, labels


def read_line(data_im_path, silence):

    with open(data_im_path, 'r', encoding='UTF-8') as fp:
        im_inf = []
        for line in fp:
            # print(line)
            line = line.strip()
            if line != '':
                im_inf.append(line)
    if silence:
        print(f"Total examples")

    # print(im_inf)

    return im_inf


def get_para_at_targets(reviews, sents, labels, im_inf):
    inputs = []
    targets = []
    for i, label in enumerate(labels):
        all_at_sentences = []
        for quad in label:
            at, ac, sp, ot = quad
            if at == 'none' or at == 'NULL':
                at = 'it'
            
            # if at == 'none' or at == 'NULL':
            #     at = 'null'

            # at = f"aspect term: {at}"
            all_at_sentences.append(at)

        temp = "Given the text: " + reviews[i] + ", what are the aspect terms in it ?" 
        inputs.append(temp.split())
        targets.append(' [SSEP] '.join(all_at_sentences))
 
    return inputs,targets

def get_para_aesc_targets(reviews, sents, labels, im_inf):
    inputs = []
    targets = []

    for i, label in enumerate(labels):
        all_tw_sentences = []
        for quad in label:
            at, ac, sp, ot = quad
            man_ot = sentword2opinion[sp]   # 'positive' -> 'great'
            if at == 'none' or at == 'NULL':
                at = 'it'

            # if at == 'none' or at == 'NULL':
            #     at = 'null'

            one_tw = f"{at} is {man_ot}"
            # one_tw = f"aspect term: {at}, sentiment polarity: {sp}"
            all_tw_sentences.append(one_tw)

        temp = "Given the text: " + reviews[i] + ", what are the aspect terms and their sentiments ?"+ " Additional information: " + im_inf[i]
        inputs.append(temp.split())
        targets.append(' [SSEP] '.join(all_tw_sentences))

    return inputs,targets

def get_para_tasd_targets(reviews, sents, labels, im_inf):
    inputs = []
    targets = []

    for i, label in enumerate(labels):
        all_tri_sentences = []
        for quad in label:
            at, ac, sp = quad
            ac = ' '.join(ac.lower().split('#'))
            man_ot = sentword2opinion[sp]   # 'positive' -> 'great'
            if at == 'none' or at == 'NULL':
                at = 'it'

            # if at == 'none' or at == 'NULL':
            #     at = 'null'


            one_tri = f"{at} is {man_ot} indicates {ac} is {man_ot}"
            # one_tri = f"aspect term: {at}, sentiment polarity: {sp}, aspect category: {ac}"
            all_tri_sentences.append(one_tri)

        temp = "Given the text: " + reviews[i] + ", what are the aspect terms, sentiments and categories ?"+ " Additional information: " + im_inf[i]
        inputs.append(temp.split())
        targets.append(' [SSEP] '.join(all_tri_sentences))

    return inputs,targets

def get_para_aste_targets(reviews, sents, labels, im_inf):
    inputs = []
    targets = []

    for i, label in enumerate(labels):
        all_tri_sentences = []
        for quad in label:
            at, ac, sp, ot = quad
            ac = ' '.join(ac.lower().split('#'))
            man_ot = sentword2opinion[sp]   # 'positive' -> 'great'
            if at == 'none' or at == 'NULL': 
                at = 'it'

            # if at == 'none' or at == 'NULL':
            #     at = 'null'

            if ot == 'none' or ot == 'NULL':
                ot = 'null'

            # one_tri = f"{at} is {ot} means it is {man_ot}"
            one_tri = f"{at} is {ot} indicates it is {man_ot}"
            # one_tri = f"aspect term: {at}, opinion term: {ot}, sentiment polarity: {sp}"
            all_tri_sentences.append(one_tri)

        temp = "Given the text: " + reviews[i] + ", what are the aspect terms, opinion terms and sentiments ?"+ " Additional information: " + im_inf[i]
        inputs.append(temp.split())
        targets.append(' [SSEP] '.join(all_tri_sentences))

    return inputs,targets

def get_para_asqp_targets(reviews, sents, labels, im_inf):
    """
    Obtain the target sentence under the paraphrase paradigm
    """
    inputs = []
    targets = []
    
    for i, label in enumerate(labels):
        all_quad_sentences = []
        for quad in label:
            at, ac, sp, ot = quad
            ac = ' '.join(ac.lower().split('#'))
            man_ot = sentword2opinion[sp]  # 'POS' -> 'good'    

            if at == 'none' or at == 'NULL': 
                at = 'it'

            # if at == 'none' or at == 'NULL':
            #     at = 'null'

            if ot == 'none' or ot == 'NULL':
                ot = 'null'
            

            # one_quad_sentence = f"{at} is {ot} means {ac} is {man_ot}"
            one_quad_sentence = f"{at} is {ot} indicates {ac} is {man_ot}"

            # one_quad_sentence = f"aspect term: {at}, opinion term: {ot}, sentiment polarity: {sp}, aspect category: {ac}"
            all_quad_sentences.append(one_quad_sentence)

        temp = "Given the text: " + reviews[i] + ", what are the aspect terms, opinion terms, sentiments and categories ?"+ " Additional information: " + im_inf[i]
        inputs.append(temp.split())
        targets.append(' [SSEP] '.join(all_quad_sentences))

    return inputs,targets


def f_get_transformed_io(data_path, data_im_path, absa_task):
    sents, reviews, labels = read_line_examples_from_file(data_path, silence=True)
    inputs = []
    targets = []
    # inputs = [s.copy() for s in sents]

    # combined = list(zip(reviews, sents, labels))  
    # random.shuffle(combined)           
    # reviews, sents, labels = zip(*combined)  
    
    im_inf = read_line(data_im_path, silence=True) 
    
    if absa_task == "tasd":
      inputs3,targets3 = get_para_tasd_targets(reviews, sents, labels, im_inf)
      inputs.extend(inputs3)
      targets.extend(targets3)

    else:
      # inputs1,targets1 = get_para_at_targets(reviews, sents, labels)
      inputs1,targets1 = get_para_at_targets(reviews, sents, labels, im_inf)
      inputs.extend(inputs1)
      targets.extend(targets1)

      inputs2,targets2 = get_para_aesc_targets(reviews, sents, labels, im_inf)
      inputs.extend(inputs2)
      targets.extend(targets2)



      inputs4,targets4 = get_para_aste_targets(reviews, sents, labels, im_inf)
      inputs.extend(inputs4)
      targets.extend(targets4)

      inputs5,targets5 = get_para_asqp_targets(reviews, sents, labels, im_inf)
      # targets5 = get_para_asqp_targets(reviews, sents, labels)
      inputs.extend(inputs5)
      targets.extend(targets5)

    # inputs = inputs*3
    # targets = targets*3

    # combined = list(zip(inputs, targets))  
    # random.shuffle(combined)           
    # inputs, targets = zip(*combined)   
  
    return inputs, targets

def get_transformed_io(task, data_path, data_im_path):
    """
    The main function to transform input & target according to the task
    """
    sents, reviews, labels = read_line_examples_from_file(data_path, silence=True)

    im_inf = read_line(data_im_path, silence=True) 

    # the input is just the raw sentence
    # inputs = [s.copy() for s in sents]

    if task == 'at':
        inputs,targets = get_para_at_targets(reviews, sents, labels, im_inf)
    elif task == 'aesc':
        inputs,targets = get_para_aesc_targets(reviews, sents, labels, im_inf)
    elif task == 'tasd':
        inputs,targets = get_para_tasd_targets(reviews, sents, labels, im_inf)
    elif task == 'aste':
        inputs,targets = get_para_aste_targets(reviews, sents, labels, im_inf)
    elif task == 'asqp':
        inputs,targets = get_para_asqp_targets(reviews, sents, labels, im_inf)
        # targets = get_para_asqp_targets(reviews, sents, labels)
    else:
        raise NotImplementedError

    return inputs, targets

class ABSADataset(Dataset):
    def __init__(self, tokenizer, data_dir, absa_task, data_count, data_type, max_len=128):


        if data_type == 'train':
            self.data_path = f'./01_augmentations/fs_examples/{absa_task}/{data_dir}/fs_{data_count}/aug.txt'   
            self.data_im_path = f'./01_augmentations/fs_examples/{absa_task}/{data_dir}/fs_{data_count}/aug_im.txt'   
        elif data_type == 'dev':
            self.data_path = f'../zero-shot-absa-quad/datasets/{absa_task}/{data_dir}/{data_type}.txt'
            self.data_im_path = f'./02_dataset_augmentations/{absa_task}/{data_dir}/{data_type}_im.txt'   
        else:
            self.data_path = f'../zero-shot-absa-quad/datasets/{absa_task}/{data_dir}/{data_type}.txt'
            self.data_im_path = f'./02_dataset_augmentations/{absa_task}/{data_dir}/{data_type}_im.txt'   

        self.absa_task = absa_task
        self.data_type = data_type
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.data_count = data_count
        self.inputs = []
        self.targets = []
        self._build_examples()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

        return {"source_ids": source_ids, "source_mask": src_mask, 
                "target_ids": target_ids, "target_mask": target_mask}

    def _build_examples(self):

        
        if self.data_type == 'train' or self.data_type == 'dev':
            inputs, targets = f_get_transformed_io(self.data_path, self.data_im_path, self.absa_task)
            if self.data_type == 'dev':
                inputs = inputs[:64]
                targets = targets[:64]
        else:
            inputs, targets = get_transformed_io(self.absa_task, self.data_path, self.data_im_path) 

        for i in range(len(inputs)):
            # change input and target to two strings
            input = ' '.join(inputs[i])
            target = targets[i]

            tokenized_input = self.tokenizer.batch_encode_plus(
              [input], max_length=self.max_len, padding="max_length",
              truncation=True, return_tensors="pt"
            )
            tokenized_target = self.tokenizer.batch_encode_plus(
              [target], max_length=self.max_len, padding="max_length",
              truncation=True, return_tensors="pt"
            )

            self.inputs.append(tokenized_input)
            self.targets.append(tokenized_target)