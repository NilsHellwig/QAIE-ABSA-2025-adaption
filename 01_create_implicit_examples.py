import os
import pprint
from tqdm import tqdm
from llm import LLM

model = LLM()

TASKS = ["tasd", "asqp"]
N_SHOTS = [10, 50]
DATASETS = ["rest15", "rest16", "flightabsa", "hotels", "coursera"]


def read_line_examples_from_file(data_path, silence):
    reviews, sents, labels = [], [], []
    with open(data_path, "r", encoding="UTF-8") as fp:
        words, labels = [], []
        for line in fp:
            line = line.strip()
            if line != "":
                words, tuples = line.split("####")
                reviews.append(words)
                sents.append(words.split())
                labels.append(eval(tuples))
    if silence:
        print(f"Total examples = {len(sents)}")
    return sents, reviews, labels


def ask_question(question, x):
    if x == 0:
        prompt = f"Q: {question}\nA:"
    if x == 1:
        prompt = f"Q: {question1}\nA: {answer1}\nQ: {question}\nA:"
    if x == 2:
        prompt = f"Q: {question1}\nA: {answer1}\nQ: {question2}\nA: {answer2}\nQ: {question}\nA:"

    pred_ok = False
    while pred_ok != True:
       try:
         answer = model.predict(prompt, seed=0, stop=[], temperature=0.8)[0]
         pred_ok = True
       except:
          pass
    return answer


def do_augmentation(data_path, datai_path):
    sents, reviews, labels = read_line_examples_from_file(data_path, silence=True)

    answers = []
    i = 0

    for review in reviews:

        question1 = f"""Given the text "{reviews[i]}", what is your understanding of the text? Keep your answers short."""
        answer1 = ask_question(question1, 0)
        question2 = f"""The text "{reviews[i]}" is a comment from the restaurant field. answer briefly. "{answer1}" What does the sentence imply ? answer briefly."""
        answer2 = ask_question(question2, 0)
        answer2 = (
            answer2.strip()
            .replace("A: ", "")
            .replace("Q: ", "")
            .replace("**", "")
            .replace("\n", "")
        )
        answers.append(answer2)
        print(f"Answer {i}", answer2)

        i = i + 1
        # if i == 8:
        #     break

    with open(datai_path, "w") as f:
        for item in answers:
            f.write("%s\n" % item)


DATASET_TYPE = ["train", "dev", "test"]        

for task in TASKS:
    for dataset_type in DATASET_TYPE:
        for dataset in DATASETS:
            data_path = f"../zero-shot-absa-quad/datasets/{task}/{dataset}/{dataset_type}.txt"
            datai_path = f"./02_dataset_augmentations/{task}/{dataset}/{dataset_type}_aug.txt"
            # create datai directories if not exist
            datai_dir = os.path.dirname(datai_path)
            if not os.path.exists(datai_dir):
                os.makedirs(datai_dir)
                
            do_augmentation(data_path, datai_path)
            

for task in TASKS:
    for n_shot in N_SHOTS:
        for dataset in DATASETS:
            data_path = f"../zero-shot-absa-quad/fs_examples/{task}/{dataset}/fs_{n_shot}/examples.txt"
            datai_path = f"./01_augmentations/fs_examples/{task}/{dataset}/fs_{n_shot}/examples_aug.txt"
            # create datai directories if not exist
            datai_dir = os.path.dirname(datai_path)
            if not os.path.exists(datai_dir):
                os.makedirs(datai_dir)
                
            do_augmentation(data_path, datai_path)
            
    


