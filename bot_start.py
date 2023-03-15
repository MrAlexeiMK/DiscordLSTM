import os
import getopt
import sys
import discord
import csv
import pathlib
from bot_train import *

# TensorFlow model
MODEL = None

# For converting word to token
TOKENIZER = None


def load(modelPath):
    global MODEL, TOKENIZER

    import tensorflow as tf

    MODEL = tf.keras.models.load_model(modelPath)
    MODEL.summary()

    with open(modelPath + 'dict.pickle', 'rb') as handle:
        TOKENIZER = pickle.load(handle)

    vocab_size = len(TOKENIZER.word_index)
    print("[INFO] Loaded model contains " + str(vocab_size) + " words")
    print("[INFO] Model successfully loaded!")


def predict(input_text):
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    num_words = 50
    max_repeats = 3

    output_text = input_text
    current_repeats = 1
    last_word = None

    # reverse map to get predicted word
    reverse_word_map = dict(map(reversed, TOKENIZER.word_index.items()))

    while num_words > 0:
        encoded_input = pad_sequences(TOKENIZER.texts_to_sequences(
            NLP.extract_sentences_normalized(output_text)), padding='post', maxlen=QUESTION_MAX_WORDS)

        probs_output = MODEL.predict(encoded_input)[0]

        index = np.argmax(probs_output[1:]) + 1

        if index != 0:
            word = reverse_word_map[index]
            if word == "$":
                break

            if last_word != None:
                if last_word == word:
                    current_repeats += 1
                else:
                    current_repeats = 1

            if current_repeats > max_repeats:
                break

            output_text += " " + word
            last_word = word

        num_words -= 1

    return output_text


def is_not_valid(msg):
    return '<' in msg or '>' in msg or '$' in msg or "http" in msg


def run(token, modelPath):
    intents = discord.Intents.all()
    client = discord.Client(intents=intents)
    script_dir = pathlib.Path(__file__).parent.absolute()
    os.chdir(script_dir)

    load(modelPath)

    @client.event
    async def on_message(message):
        if message.author == client.user:
            return

        if message.content.startswith('/lstm'):
            try:
                question = ' '.join(message.content.split(' ')[1:])
                answer = predict(question)
                print(f"[{question}] -> [{answer}]")
                await message.channel.send(answer)
            except:
                pass

    client.run(token)


def start_help():
    print("[INFO] python bot_start.py --token <discordBotToken> --model <modelPath>")
    sys.exit()


def main(argv):
    token = None
    modelPath = None

    opts, args = getopt.getopt(argv, "ht:m:", ["token=", "model="])
    for opt, arg in opts:
        if opt == '-h':
            start_help()
        elif opt in ("-t", "--token"):
            token = arg
        elif opt in ("-m", "--model", "--modelPath"):
            modelPath = arg
            if modelPath[-1] != '/':
                modelPath += '/'

    cancelled = False
    if token == None:
        print("[ERROR] You need to provide discord-bot token")
        cancelled = True
    if modelPath == None:
        print("[ERROR] You need to provide path to model like 'weights/model1/'")
        cancelled = True
    
    if cancelled == True:
        start_help()

    run(token, modelPath)

if __name__ == "__main__":
    main(sys.argv[1:])