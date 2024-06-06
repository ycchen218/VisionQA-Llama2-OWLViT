import os
import json
import _pickle as cPickle
import torch
from tqdm import tqdm
from zero_short import ViTLLame13B
import string
from nltk.corpus import wordnet as wn
import nltk
nltk.download('wordnet')
device = "cuda" if torch.cuda.is_available() else "cpu"
# Accuracy 0.5222889155662265

def remove_punctuation(sentence):
    return sentence.translate(str.maketrans('', '', string.punctuation))

def find_same_word(word):
    same_word_set = wn.synsets(word)
    same_word_set = [sw.lemma_names() for sw in same_word_set]
    same_word_set = [item.lower() for sublist in same_word_set for item in sublist]
    same_word_set = list(dict.fromkeys(same_word_set))
    return same_word_set
def is_substring_present(sentence1, sentence2):
    sentence1 = remove_punctuation(sentence1)
    sentence2 = remove_punctuation(sentence2)
    for s1_word in sentence1.split():
        s1_synsets = wn.synsets(s1_word)
        if s1_synsets:
            s1_synset = s1_synsets[0]  # Consider only the first synset
            for s2_word in sentence2.split():
                similarity = 0
                s2_synsets = wn.synsets(s2_word.lower())
                if s2_synsets:
                    s2_synset = s2_synsets[0]  # Consider only the first synset
                    similarity = wn.path_similarity(s1_synset, s2_synset)
                if similarity and similarity >= 0.5:  # Check if similarity is not None
                    return True
    return False

def validation(dataroot, mode,imgs_path):
    model = ViTLLame13B(device=device)
    ans2label_path = os.path.join(dataroot, 'cache', 'trainval_ans2label.pkl')
    ans2label = cPickle.load(open(ans2label_path, 'rb'))
    json_path = os.path.join(dataroot, f'{mode}_balanced_questions.json')
    acc = []
    with open(json_path, 'r') as f:
        data = list(json.load(f).values())
    for item in tqdm(data):
        img_path = os.path.join(imgs_path, item['imageId'] + '.jpg')
        # img = cv2.imread(img_path)
        if item["answer"] in ans2label.keys() and os.path.exists(img_path):
            question_sent = item['question']
            ans = item['answer']
            pred_ans = model(img_path=img_path, question=question_sent)
            score = is_substring_present(ans, pred_ans)
            acc.append(score)
    accuracy = sum(acc) / len(acc)
    return accuracy



if __name__ == '__main__':
    imgs_path = os.path.join('GQA_data/GQA/allImages', 'images')
    # 測試範例
    acc = validation(dataroot='../data/gqa', mode='testdev', imgs_path=imgs_path)
    print('Accuracy', acc)
