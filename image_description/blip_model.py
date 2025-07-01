
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, GPT2LMHeadModel, GPT2Tokenizer
from nltk.tokenize import sent_tokenize
from PIL import Image

# Загружаем модель BLIP один раз
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# Загружаем GPT-2 для генерации дополнительных предложений
gpt_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


def generate_additional_sentences(base_text, num_sentences=3):
    # Преобразуем текст в токены для GPT-2
    inputs = gpt_tokenizer.encode(base_text, return_tensors="pt").to(device)

    # Генерация дополнительных предложений с использованием beam search
    generated_text = gpt_model.generate(
        inputs,
        max_length=100,
        num_beams=5,  # Используем beam search для генерации нескольких вариантов
        num_return_sequences=min(num_sentences, 3),  # Ограничиваем до 5, чтобы не было больше beams
        no_repeat_ngram_size=2,
        temperature=0.7,
        top_k=50,
        top_p=0.95
    )

    # Декодируем и получаем предложения
    sentences = []
    for seq in generated_text:
        decoded = gpt_tokenizer.decode(seq, skip_special_tokens=True)
        new_sentences = sent_tokenize(decoded)
        sentences.extend(new_sentences)

    return sentences


def get_blip_description(image_path):
    raw_image = Image.open(image_path).convert('RGB')

    inputs = processor(raw_image, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)

    # Разделим сгенерированное описание на предложения
    sentences = sent_tokenize(caption)

    # Если предложений меньше 10, генерируем дополнительные
    while len(sentences) < 3:
        # Генерируем дополнительные уникальные предложения
        additional_sentences = generate_additional_sentences(caption, num_sentences=3 - len(sentences))
        sentences.extend(additional_sentences)

    # Возвращаем первые 10 предложений
    return ' '.join(sentences[:3])
