import torch
from transformers import AutoProcessor, OwlViTForObjectDetection
from transformers import AutoTokenizer, AutoModelForCausalLM

import warnings
warnings.filterwarnings("ignore")
import string
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import cv2
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np


device = "cuda" if torch.cuda.is_available() else "cpu"


class ViTLLame13B():
    def __init__(self, yolo_path, device):

        self.llm_model = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-13B-chat-GPTQ",
                                                              device_map="auto",
                                                              trust_remote_code=False,
                                                              revision="main").to(device)

        self.llm_tokenizer = AutoTokenizer.from_pretrained("TheBloke/Llama-2-13B-chat-GPTQ", use_fast=True)
        self.llm_model.eval()
        self.vit_processor = AutoProcessor.from_pretrained("google/owlvit-base-patch32")
        self.vit_model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
        self.vit_model.eval()

        self.sys_prompt = " <<SYS>>You are an assistant tasked with answering questions. Your answers must be brief. Each question will provide details about the objects and their locations in the image. Use this information to answer the questions, and do not include any information that is not provided in the question.<</SYS>>"

        self.stop_words = set(stopwords.words('english'))
        self.yolo = YOLO(yolo_path).to(device)
        self.max_det = 10

    def remove_punctuation(self,sentence):
        return sentence.translate(str.maketrans('', '', string.punctuation))
    def calculate_iou(self, box1, box2):
        intersection = [
            max(box1[0], box2[0]),
            max(box1[1], box2[1]),
            min(box1[2], box2[2]),
            min(box1[3], box2[3])
        ]
        if intersection[2] <= intersection[0] or intersection[3] <= intersection[1]:
            return 0
        intersection_area = (intersection[2] - intersection[0]) * (intersection[3] - intersection[1])
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - intersection_area

        iou = intersection_area / union_area
        return iou

    def remove_overlapping_labels(self, labels_list, boxes_list, iou_threshold=0.8):
        keep = [True] * len(labels_list)
        for i in range(len(labels_list)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(labels_list)):
                if labels_list[i] == labels_list[j] and self.calculate_iou(boxes_list[i],
                                                                           boxes_list[j]) > iou_threshold:
                    keep[j] = False

        new_labels_list = [labels_list[i] for i in range(len(labels_list)) if keep[i]]
        new_boxes_list = [boxes_list[i] for i in range(len(boxes_list)) if keep[i]]

        return new_labels_list, new_boxes_list

    def add_box_position_to_labels(self, sentence, labels_list, boxes_list):
        result_sentence = sentence
        used_indices = []

        for i, (label, box) in enumerate(zip(labels_list, boxes_list)):
            pattern = rf'\b{label}\b'
            if re.search(pattern, result_sentence):
                box_pos = f"at ({box[0]}, {box[1]}, {box[2]}, {box[3]})"
                result_sentence = re.sub(pattern, f"{label} {box_pos}", result_sentence, count=1)
                used_indices.append(i)

        additional_info_parts = []
        for i, (label, box) in enumerate(zip(labels_list, boxes_list)):
            if i not in used_indices:
                additional_info_parts.append(f"a {label} at ({box[0]}, {box[1]}, {box[2]}, {box[3]})")

        if additional_info_parts:
            additional_info = "there is " + " and ".join(additional_info_parts)
            result_sentence += " " + additional_info
        result_sentence += "."
        return result_sentence

    def preprocess_image(self, img, size):
        transform = LetterBox(size, True, stride=32)
        img = transform(image=img)
        return img

    def extract_detection_info(self, results, text, index=0):
        boxes, scores, labels = results[index]["boxes"], results[index]["scores"], results[index]["labels"]

        boxes_list = []
        scores_list = []
        labels_list = []

        for box, score, label in zip(boxes, scores, labels):
            box = [int(i) for i in box.tolist()]
            boxes_list.append(box)
            scores_list.append(round(score.item(), 3))
            labels_list.append(text[label])

        return boxes_list, scores_list, labels_list

    def owl_vit_det(self, texts, img, threshold=0.1):
        inputs = self.vit_processor(text=texts, images=img, return_tensors="pt")
        outputs = self.vit_model(**inputs)
        target_sizes = torch.Tensor([img.shape[:2]])
        # target_sizes = torch.Tensor([img.size[::-1]])
        results = self.vit_processor.post_process_object_detection(outputs=outputs, threshold=threshold,
                                                                   target_sizes=target_sizes)
        return results

    def llama2_answer(self, prompt, max_length=256):
        prompt_template = f'''[INST]{self.sys_prompt} {prompt}[/INST]'''
        tokens = self.llm_tokenizer(prompt_template, return_tensors='pt').input_ids.to(device)
        generation_output = self.llm_model.generate(
            tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            max_new_tokens=max_length
        )
        response = self.llm_tokenizer.decode(generation_output[0])
        return response

    def normalize_boxes(self, xywh, img):
        assert xywh.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {xywh.shape}"

        # Get image dimensions
        h_img, w_img = img.shape[:2]

        y = torch.empty_like(xywh) if isinstance(xywh, torch.Tensor) else np.empty_like(xywh)

        x_center = xywh[..., 0]
        y_center = xywh[..., 1]
        w = xywh[..., 2]
        h = xywh[..., 3]

        y[..., 0] = x_center / w_img
        y[..., 1] = y_center / h_img
        y[..., 2] = w / w_img
        y[..., 3] = h / h_img

        return y

    def denormalize_boxes(self, normalized_boxes, img):
        assert normalized_boxes.shape[
                   -1] == 4, f"input shape last dimension expected 4 but input shape is {normalized_boxes.shape}"

        # Get image dimensions
        h_img, w_img = img.shape[:2]

        denormalized_boxes = torch.empty_like(normalized_boxes) if isinstance(normalized_boxes,
                                                                              torch.Tensor) else np.empty_like(
            normalized_boxes)

        x_center = normalized_boxes[..., 0] * w_img
        y_center = normalized_boxes[..., 1] * h_img
        w = normalized_boxes[..., 2] * w_img
        h = normalized_boxes[..., 3] * h_img

        denormalized_boxes[..., 0] = x_center - w / 2  # x1
        denormalized_boxes[..., 1] = y_center - h / 2  # y1
        denormalized_boxes[..., 2] = x_center + w / 2  # x2
        denormalized_boxes[..., 3] = y_center + h / 2  # y2

        return denormalized_boxes

    def yolo_detect(self, raw_img):
        img = self.preprocess_image(raw_img, size=640)
        results = self.yolo(img, verbose=False, max_det=self.max_det, conf=0.1)[0]
        classes_name = self.yolo.names
        pred_cls = results.boxes.cls.detach().cpu().numpy().astype(int)
        pred_cls_names = [classes_name[idx] for idx in pred_cls]
        pred_scores = results.boxes.conf.detach().cpu().numpy()
        boxes_pos = self.normalize_boxes(results.boxes.xywh, img)
        boxes_pos = self.denormalize_boxes(boxes_pos, raw_img).detach().cpu().numpy().astype(int).tolist()
        return boxes_pos, pred_scores, pred_cls_names


    def parse_input(self, input_str, labels_list, boxes_list):
        draw_boxes = []
        draw_labels = []
        input_str = self.remove_punctuation(input_str.lower()).split()
        for l, b in zip(labels_list, boxes_list):
            if l in input_str:
                draw_boxes.append(b)
                draw_labels.append(l)
        return draw_labels, draw_boxes

    def show_box(self, box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(
            plt.Rectangle((int(x0), int(y0)), int(w), int(h), edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
        )

    def show_boxes_and_labels_on_image(self, input_str, labels_list, boxes_list, raw_image):
        labels, boxes = self.parse_input(input_str, labels_list, boxes_list)
        plt.figure(figsize=(10, 10))
        plt.imshow(raw_image)
        for i, box in enumerate(boxes):
            self.show_box(box, plt.gca())
            plt.text(
                x=box[0],
                y=box[1] - 12,
                s=f"{labels[i]}",
                c="beige",
                path_effects=[pe.withStroke(linewidth=4, foreground="darkgreen")],
            )
        plt.axis("on")
        plt.show()

    def __call__(self, img_path, question, show_img=False):
        tokenized_question = word_tokenize(question)
        filtered_question_sent = ' '.join(
            [w for w in tokenized_question if not w.lower() in self.stop_words]).split()
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        text_results = self.owl_vit_det(texts=filtered_question_sent, img=img, threshold=0.1)
        text_boxes_list, _, text_labels_list = self.extract_detection_info(text_results, filtered_question_sent,
                                                                           index=0)
        yolo_boxes_list, _, yolo_labels_list = self.yolo_detect(img)
        labels_list = text_labels_list[:self.max_det] + yolo_labels_list[:self.max_det]
        boxes_list = text_boxes_list[:self.max_det] + yolo_boxes_list[:self.max_det]
        labels_list, boxes_list = self.remove_overlapping_labels(labels_list, boxes_list, iou_threshold=0.5)
        processed_question = self.add_box_position_to_labels(question, labels_list, boxes_list)
        answer = self.llama2_answer(processed_question)
        answer = answer.split("[/INST]")[1].split("</s>")[0].split("2. ")[0]
        if show_img:
            self.show_boxes_and_labels_on_image(answer, labels_list, boxes_list, img)
        return answer


if __name__ == "__main__":
    img_path = 'test.jpg'
    a = cv2.imread(img_path)
    question_sent = 'What is in the photo?'
    full_model = ViTLLame13B(yolo_path="yolo_weight.pt", device=device)

    while True:
        question_sent = input("Text me the question ('q' to quit)：")
        if question_sent.lower() == 'q':
            print("Quit VQA")
            break
        print("Question: ", question_sent)
        answer = full_model(img_path, question_sent, show_img=True)
        print("Answer: ", answer)
