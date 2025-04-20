import os, re
import pprint
from tqdm import tqdm
import time
from llm import LLM

model = LLM(base_model="gemma3:27b")

senot = {"positive": "negative", "negative": "positive"}


def read_line_examples_from_file(data_path, silence):
    """
    Read data from file, each line is: sent####labels
    Return List[List[word]], List[Tuple]
    """
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
        prompt = f"Q: {question}\nA:"  # 格式化问题

    pred_ok = False
    temp = 0.8
    while pred_ok != True:
        anser = None
        try:
            # raise error if "**Output:**"" "Q:" "A:" in answer
            answer = model.predict(prompt, seed=0, stop=[], temperature=temp)[0]
            answer = answer.strip()
            answer = answer.replace("\n", "")

            if re.search(r"\*\*<\w+>:\*\*", answer):
                raise ValueError("Answer contains **<word>:**")
            
            if "Q:" in answer:
                raise ValueError("Answer contains Q:")
            if "A:" in answer:
                raise ValueError("Answer contains A:")
            # if "\n" in answer:
            #     raise ValueError("Answer contains \\n")
            if "###" in answer:
                raise ValueError("Answer contains ###")
            
            pred_ok = True
        except:
            print("error!", answer)
            temp += 0.1
            pass
    return answer


def at_sm(at, review):

    question = f"""<Input> Task Description: You can generate a new smooth text by following the Implementation Details based on the given texts and words. Give you these elements: Original Text, Replaced Word, Implementation Details, An input and output example, Principle, Tip. Finally, output the new text with the replacement.
                            Original Text: "{review}".
                            Replaced Word: "{at}".
                            Implementation Details: Finding a similar word or phrase with ''Replaced Word'' based on its actual meaning in the text ''Original Text'' and replacing the ''Replaced Word'' in the Text ``Original Text'' with this found word. 
                            An input and output example: <Input> Task Description: You can generate ..., Original Text: good drink, Replaced Word: drink, Implementation Details: Finding a similar word ..., Principle: Except for the word ..., Tip: If you can't ..., <Output> good beverage.
                            Principle: Except for the word in the replaced position in the input text, which can be changed, the word in the other positions in the text remains unchanged.
                            Tip: If you can't choose a suitable word, look up synonyms or related words and check that the replacement text makes sense. Always make sure that the replacement word matches the grammar and context of the original text.
                    <Output>"""
    answer = ask_question(question, 0)
    # print("Answer at_sm:\n", answer, "\n#######\n")

    return answer


def ot_sm(ot, review):
    question = f"""<Input> Task Description: You can generate a new smooth text by following the Implementation Details based on the given texts and words. Give you these elements: Original Text, Replaced Word, Implementation Details, An input and output example, Principle, Tip. Finally, output the new text with the replacement.
                            Original Text: "{review}".
                            Replaced Word: "{ot}".
                            Implementation Details: Finding a similar word or phrase with ''Replaced Word'' based on its actual meaning in the text ''Original Text'' and replacing the ''Replaced Word'' in the Text ``Original Text'' with this found word. 
                            An input and output example: <Input> Task Description: You can generate ..., Original Text: good drink, Replaced Word: good, Implementation Details: Finding a similar word ..., Principle: Except for the word ..., Tip: If you can't ..., <Output> great drink.
                            Principle: Except for the word in the replaced position in the input text, which can be changed, the word in the other positions in the text remains unchanged.
                            Tip: If you can't choose a suitable word, look up synonyms or related words and check that the replacement text makes sense. Always make sure that the replacement word matches the grammar and context of the original text.
                    <Output>"""
    answer = ask_question(question, 0)
    # print("Answer ot_sm:\n", answer,"\n#######\n")

    return answer


def ot_op(ot, review):

    question = f"""<Input> Task Description: You can generate a new smooth text by following the Implementation Details based on the given texts and words. Give you these elements: Original Text, Replaced Word, Implementation Details, An input and output example, Principle, Tip. Finally, output the new text with the replacement.
                            Original Text: "{review}".
                            Replaced Word: "{ot}".
                            Implementation Details: Finding a opposite word or phrase with ''Replaced Word'' based on its actual meaning in the text ''Original Text'' and replacing the ''Replaced Word'' in the Text ``Original Text'' with this found word. 
                            An input and output example: <Input> Task Description: You can generate ..., Original Text: good drink, Replaced Word: good, Implementation Details: Finding a opposite word ..., Principle: Except for the word ..., Tip: If you can't ..., <Output> bad drink.
                            Principle: Except for the word in the replaced position in the input text, which can be changed, the word in the other positions in the text remains unchanged.
                            Tip: If you can't choose a suitable word, look up synonyms or related words and check that the replacement text makes sense. Always make sure that the replacement word matches the grammar and context of the original text.
                    <Output>"""
    answer = ask_question(question, 0)
    # print("Answer ot_op:\n", answer,"\n#######\n")

    return answer


def find_label(sentence, review):

    question = f"""<Input> Task Description: You can find out the difference between the given two texts by Implementation Details. Give you these elements: Original Text, Augmentation Text, Implementation Details, An input and output example, Principle, Tip. Finally output difference between the Original Text and the Augmentation Text.
                            Original Text: "{review}".
                            Augmented Text: "{sentence}".
                            Implementation Details: The text generated when the original text ''Original Review Text'' is replaced with the specified string is ''Augmented Review Text''. Find the string used to replace the specified string.
                            An input and output example: <Input> Task Description: You can find ..., Original Text: good drink, Replaced Word: good beverage, Implementation Details: The text generated..., Principle: Except for the word.., Tip: If you can't..., <Output> beverage
                            Principle: The output string must be found in the text ''Augmented Review Text''.
                            Tip: If you can't find this label, you can gradually narrow it down by elimination. You can use context clues to infer the possible string.
                    <Output>"""
    answer = ask_question(question, 0)
    # print("Answer find_label:\n", answer,"\n#######\n")

    return answer


def run_augmentation(dataset, n_shot, task="tasd"):
    data_path = (
        f"../zero-shot-absa-quad/fs_examples/{task}/{dataset}/fs_{n_shot}/examples.txt"
    )
    data_aug_path = (
        f"./01_augmentations/fs_examples/{task}/{dataset}/fs_{n_shot}/aug.txt"
    )

    sents, reviews, labels = read_line_examples_from_file(data_path, silence=True)

    text = []
    for i, (review, label) in enumerate(zip(reviews, labels)):
        try:
            print(text[i-1])
        except:
            pass
        if len(label) == 1:
            for quad in label:
                at, ac, sp = quad
                text.append(review + "####" + str([(at, ac, sp)]))

                if at != "none":
                    ats_sentence = at_sm(at, review)
                    new_at = find_label(ats_sentence, review)
                    if ats_sentence is not None and new_at in ats_sentence:
                        text.append(ats_sentence + "####" + str([(new_at, ac, sp)]))

        else:
            label_new = []
            for qua in label:
                label_new.append(list(qua))

            text.append(review + "####" + str(label))
            at, ac, sp = label_new[0]
            senten = review  # 初始化
            label11_lis = label_new  # 初始化
            if at != "none":
                ats_sentence = at_sm(at, review)
                new_at = find_label(ats_sentence, review)

                label11 = []
                label11_lis = []
                for qu in label_new:
                    my_list = []
                    if qu[0] == at:
                        my_list = [new_at, qu[1], qu[2]]
                    else:
                        my_list = qu
                    label11.append(tuple(my_list))
                    label11_lis.append(my_list)

                senten = ats_sentence
                if ats_sentence is not None and new_at in ats_sentence:
                    text.append(ats_sentence + "####" + str(label11))


            at, ac, sp = label_new[1]

            if at != "none" and at != label_new[0][0]:
                ats_sentence = at_sm(at, senten)
                new_at = find_label(ats_sentence, senten)

                label21 = []
                label21_lis = []
                for i in range(len(label11_lis)):
                    my_list = []
                    if i == 0:
                        my_list = label11_lis[i]
                    else:
                        if label11_lis[i][0] == at:
                            my_list = [
                                new_at,
                                label11_lis[i][1],
                                label11_lis[i][2],
                            ]
                        else:
                            my_list = label11_lis[i]

                    label21.append(tuple(my_list))
                    label21_lis.append(my_list)
                if ats_sentence is not None and new_at in ats_sentence:
                    text.append(ats_sentence + "####" + str(label21))


    # create dir for data_aug_path if not exist
    data_aug_dir = os.path.dirname(data_aug_path)
    if not os.path.exists(data_aug_dir):
        os.makedirs(data_aug_dir)
    # write data_aug_path
    with open(data_aug_path, "w") as f:
        for item in text:
            f.write("%s\n" % item)



# run_augmentation
TASKS = ["tasd"]
N_SHOTS = [10, 50]
DATASETS = ["rest15", "rest16", "flightabsa", "hotels", "coursera"]


for task in TASKS:
    for dataset in DATASETS:
        for n_shot in N_SHOTS:
            run_augmentation(
                dataset=dataset,
                n_shot=n_shot,
                task=task,
            )