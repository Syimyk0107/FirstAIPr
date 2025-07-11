from deep_translator import GoogleTranslator

# Пример перевода
result = GoogleTranslator(source='auto', target='ru').translate("Hello World!")
print(result)
