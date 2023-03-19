import getopt
import sys
import re
import pickle
import numpy as np
import signal
from nltk.tokenize import word_tokenize, sent_tokenize

# Max words in question
QUESTION_MAX_WORDS = 20

# TensorFlow model
MODEL = None

# Save every count of epoch
SAVE_EVERY_EPOCHS = 10

# For converting word to token
TOKENIZER = None

# Static class for preparing data
class NLP:
    @staticmethod
    def extract_tokens(text):
        tokens = [word.lower() for sent in sent_tokenize(text) for word in word_tokenize(sent)]
        return tokens
    
    @staticmethod
    def extract_sentences(text):
        return sent_tokenize(text)
    
    @staticmethod
    def extract_sentences_normalized(text):
        sentences = NLP.extract_sentences(text)
        for i in range(len(sentences)):
            tokens = NLP.extract_tokens(sentences[i])
            sentences[i] = ' '.join(tokens)
        return sentences
    
    @staticmethod
    def normalize_words(array):
        result = array.copy()
        for i in range(len(result)):
            text = re.sub('[^\w$ ]', '', result[i])
            try:
                result[i] = NLP.extract_sentences_normalized(text)[0]
            except:
                result[i] = None
        return np.array(result)
    
def save(modelPath):
    MODEL.save(modelPath)

    with open(modelPath + 'dict.pickle', 'wb') as handle:
        pickle.dump(TOKENIZER, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
def create(modelPath, vocabSize):
    global MODEL

    print("[INFO] Model is creating...")
    import tensorflow as tf

    MODEL = tf.keras.Sequential([
        # [26, 125, 1, ..., 0, 0, 0] * QUESTION_MAX_WORDS
        tf.keras.layers.Embedding(vocabSize, 256, input_length=QUESTION_MAX_WORDS),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(64),
        # [0.1, 0.15, 0.35, ..., 0.02, 0.13] * vocab_size
        tf.keras.layers.Dense(vocabSize, activation='softmax')
    ])
    MODEL.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])

    MODEL.summary()

    save(modelPath)

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

def train(modelPath, dataPath, extractLimit, epochs, trains):
    global MODEL, TOKENIZER
    
    loaded = False
    try:
        load(modelPath)
        loaded = True
    except:
        print("[INFO] Model doesn't exist at path '" + modelPath + "'. It will be created soon")

    import pandas as pd
    from IPython.display import display

    df = pd.read_csv(dataPath, delimiter='$')
    df = df.dropna(how='any',axis=0) 
    display(df)

    dataset = df.to_numpy()
    pred_questions = NLP.normalize_words(dataset[:, 0].tolist())
    pred_answers = NLP.normalize_words([s + " $" for s in dataset[:, 1].tolist()])

    questions = []
    answers = []

    for i in range(len(pred_questions)):
        pq = pred_questions[i]
        pa = pred_answers[i]
        if pq is None or pa is None:
            continue
        
        words = pa.split()
        
        for i in range(len(words)):
            q = f"{pq} {' '.join(words[:i])}"
            a = words[i]
            questions.append(q)
            answers.append(a)

    for i in range(min(15, len(questions))):
        print(f"[INFO] [{questions[i]}] -> [{answers[i]}]")
    print("[INFO] ...")

    from sklearn.utils import shuffle
    questions, answers = shuffle(questions, answers)

    questions = questions[:extractLimit]
    answers = answers[:extractLimit]

    print(f"\n[INFO] Extracted: {len(questions)}")

    from tensorflow.keras.preprocessing.text import Tokenizer

    TOKENIZER = Tokenizer(filters='!"#%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    TOKENIZER.fit_on_texts(questions + answers)

    vocab_size = len(TOKENIZER.word_index) + 1
    data_size = len(questions)

    print(f"[INFO] Model contains {vocab_size} words in dictionary")

    if MODEL == None:
        create(modelPath, vocab_size)

    from tensorflow.keras.preprocessing.sequence import pad_sequences

    #Get numbers sequences by tokenizer
    Q_temp = TOKENIZER.texts_to_sequences(questions)
    A_temp = TOKENIZER.texts_to_sequences(answers)
    Q = pad_sequences(Q_temp, padding='post', maxlen=QUESTION_MAX_WORDS)
    A = pad_sequences(A_temp, padding='post', maxlen=1)

    print(f"\n[INFO] Dataset size is: {data_size}")
    print(f"[INFO] Questions shape: {Q.shape}")
    print(f"[INFO] Answers shape: {A.shape}")

    from tensorflow.keras.utils import to_categorical
    A = to_categorical(A, num_classes=vocab_size)
    print(f"[INFO] Categorical answers shape: {A.shape}\n")
    
    import tensorflow as tf
    
    # Model fit callback on each epoch
    class ModelSaving(tf.keras.callbacks.Callback):
        def __init__(self, modelPath):
            self.modelPath = modelPath
        def on_epoch_end(self, epoch, logs = None):
            if (epoch + 1) % SAVE_EVERY_EPOCHS == 0:
                save(self.modelPath)

    if loaded:
        print("[WARNING] Loaded model accuracy can be lower than expected cause of another validation test set")
        
    def exit_handler(sig, frame):
        print('\n[INFO] You pressed [Ctrl + C]. Saving the model...')
        save(modelPath)
        sys.exit(0)
        
    signal.signal(signal.SIGINT, exit_handler)

    try:
        MODEL.fit(Q, A, epochs=epochs, shuffle=True, validation_split=0.2, callbacks=[ModelSaving(modelPath)])
    except ValueError:
        print(f"[WARNING] Model will be recreated at '{modelPath}' cause of different tokenizer")
        create(modelPath, vocab_size)
        MODEL.fit(Q, A, epochs=epochs, shuffle=True, validation_split=0.2, callbacks=[ModelSaving(modelPath)])
    
    save(modelPath)

    if trains > 0:
        train(modelPath, dataPath, extractLimit, epochs, trains - 1)

def train_help():
    print('[INFO] python bot_train.py --modelPath <modelPath> --dataPath <trainDataPath> -l [extractLimit] -e [epochsCount] --trains [countTrains]')
    sys.exit()

def main(argv):
    modelPath = None
    dataPath = None
    extractLimit = 200000
    epochs = 50
    trains = 1

    opts, args = getopt.getopt(argv, "hm:d:l:e:s:",["modelPath=", "dataPath=", "limit=", "epochs=", "trains="])
    for opt, arg in opts:
        if opt == '-h':
            train_help()
        elif opt in ("-m", "--modelPath"):
            modelPath = arg
            if modelPath[-1] != '/':
                modelPath += '/'
        elif opt in ("-d", "--dataPath"):
            dataPath = arg
        elif opt in ("-l", "--limit", "--extractLimit"):
            extractLimit = int(arg)
        elif opt in ("-e", "--epochs"):
            epochs = int(arg)
        elif opt in ("-t", "--trains"):
            trains = int(arg)

    cancelled = False

    if modelPath == None:
        print("[ERROR] You need to provide path to model like 'weights/model1/'")
        cancelled = True
    if dataPath == None:
        print("[ERROR] You need to provide path to extracted discord messages '.csv' file like 'data/discord_conversation_id.csv'")
        cancelled = True

    if cancelled:
        train_help()
    
    train(modelPath, dataPath, extractLimit, epochs, trains)

if __name__ == "__main__":
    main(sys.argv[1:])