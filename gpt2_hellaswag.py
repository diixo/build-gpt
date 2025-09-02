"""
The validation set of HellaSwag has a total of 10,042 examples.
"""
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch


model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.eval()
model = model.to(device)


dataset = load_dataset("hellaswag", split="validation")


def test_10_shots():
    shots = 10
    correct = 0
    total = 0

    for i in range(len(dataset)):
        test_item = dataset[i]

        # Выбираем 10 случайных примеров из train для few-shot
        few_shot_examples = random.sample(dataset, shots)

        # Формируем prompt
        prompt = ""
        for ex in few_shot_examples:
            prompt += f"Context: {ex['ctx']} Choices: {', '.join(ex['endings'])} Answer: {ex['endings'][ex['label']]}\n"

        # Добавляем тестовый пример
        prompt += f"Context: {test_item['ctx']} Choices: {', '.join(test_item['endings'])} Answer: "

        # Получаем score для каждого варианта
        scores = score_choices(prompt, test_item['endings'])
        pred = scores.index(max(scores))

        if pred == test_item['label']:
            correct += 1
        total += 1
    accuracy = correct / total * 100
    return accuracy


def score_choices(context, choices):
    scores = []
    for choice in choices:
        text = context + " " + choice
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # Считаем log-probability только для токенов продолжения
        token_ids = inputs["input_ids"][0]
        choice_len = len(tokenizer(choice)["input_ids"])
        # берем последние choice_len токенов
        start_idx = -choice_len
        log_probs = torch.log_softmax(logits[0, start_idx-1:-1], dim=-1)
        selected_token_ids = token_ids[start_idx:]
        score = log_probs[range(choice_len), selected_token_ids].sum().item()
        scores.append(score)
    return scores


def test():
    correct = 0
    total = 0

    sz = len(dataset)
    for i in range(sz):
        item = dataset[i]
        context = item['ctx']
        choices = item['endings']
        label = item['label']

        scores = score_choices(context, choices)
        pred = scores.index(max(scores))

        if pred == label:
            correct += 1
        total += 1

        if (i + 1) % 100 == 0 or (i == 0):
            print(f"...items: {i+1} ({sz})")

    accuracy = correct / total * 100
    return accuracy


if __name__ == "__main__":

    accuracy_1 = test()
    print(f"Accuracy on HellaSwag: {accuracy_1:.2f}%")

