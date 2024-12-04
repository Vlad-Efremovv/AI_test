import numpy as np
from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields  # Измените эту строчку

app = Flask(__name__)
api = Api(app, version='1.0', title='Text Generation API',
          description='An API to generate text based on provided words.')

# Определение модели ввода
text_gen_model = api.model('TextGen', {
    'word': fields.String(required=True, description='The starting word for text generation'),
    'count': fields.Integer(required=False, description='Number of words to generate', default=100)
})

# Функция для чтения и подготовки корпуса
def crate_corpus() -> list:
    text = open("bg.txt", encoding='utf-8').read() 
    corpus = text.lower().split()
    return corpus

# Генератор пар слов из корпуса
def make_pairs(corpus: list):
    for i in range(len(corpus) - 1):
        yield (corpus[i], corpus[i + 1])
    
# Подготовка словаря для генерации текста
def prepare_dict():
    corpus = crate_corpus()
    pairs = make_pairs(corpus)
    word_dict = {}

    for word_1, word_2 in pairs:
        if word_1 not in word_dict:
            word_dict[word_1] = {}
        if word_2 not in word_dict[word_1]:
            word_dict[word_1][word_2] = 0
        word_dict[word_1][word_2] += 1
        
    return word_dict 
    
# Функция генерации текста
def generete_text(word: str, count: int = 100) -> str:
    word_dict = prepare_dict()
    chain = [word]

    for i in range(count):
        next_words = list(word_dict.get(chain[-1], {}).keys())
        
        if next_words: 
            next_word = np.random.choice(next_words)
            chain.append(next_word)
        else: 
            print(f"No available words to follow '{chain[-1]}'. Stopping generation.")
            break

    return ' '.join(chain)

# Определение ресурса для генерации текста
@api.route('/generate')
class TextGeneration(Resource):
    @api.expect(text_gen_model)
    def post(self):
        # Получаем данные из запроса
        data = request.json
        word = data.get('word')
        count = data.get('count', 100)

        # Генерируем текст
        generated_text = generete_text(word, count)
        return jsonify({'generated_text': generated_text})

if __name__ == '__main__':
    app.run(debug=True)
