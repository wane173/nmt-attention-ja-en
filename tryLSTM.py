import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Concatenate, Dot, Activation, Dense
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.model_selection import train_test_split
import json
import os
import re

# --- Configuration ---
UNITS = 128
EPOCHS = 10
BATCH_SIZE = 64
MAX_OUTPUT_LENGTH = 30
MODEL_WEIGHTS_PATH = 'nmt_attention_lstm_weights.h5'
TOKENIZER_EN_PATH = 'tokenizer_en.json'
TOKENIZER_JA_PATH = 'tokenizer_ja.json'

# --- Data Preprocessing ---
def load_data(file_path):
    if not os.path.exists(file_path):
        print(f"[Error] File not found: {file_path}")
        print("Please download 'small_parallel_enja' dataset and place train.ja/train.en.")
        return None, None
    tokenizer = Tokenizer(filters="")
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    whole_texts = ["<s> " + line.strip() + " </s>" for line in lines]
    tokenizer.fit_on_texts(whole_texts)
    sequences = tokenizer.texts_to_sequences(whole_texts)
    return sequences, tokenizer

# --- Model Building ---
def create_training_model(ja_vocab_size, en_vocab_size, units):
    # Encoder
    encoder_inputs = Input(shape=(None,), name="encoder_inputs")
    encoder_embedding = Embedding(en_vocab_size, units, mask_zero=True, name="encoder_emb")
    encoder_lstm = LSTM(units, return_sequences=True, return_state=True, name="encoder_lstm")
    enc_emb_output = encoder_embedding(encoder_inputs)
    encoder_outputs_seq, state_h, state_c = encoder_lstm(enc_emb_output)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = Input(shape=(None,), name="decoder_inputs")
    decoder_embedding = Embedding(ja_vocab_size, units, mask_zero=True, name="decoder_emb")
    decoder_lstm = LSTM(units, return_sequences=True, return_state=True, name="decoder_lstm")
    dec_emb_output = decoder_embedding(decoder_inputs)
    decoder_lstm_outputs, _, _ = decoder_lstm(dec_emb_output, initial_state=encoder_states)

    # Attention
    attention_dot1 = Dot(axes=[2, 2], name="attention_score")
    attention_softmax = Activation('softmax', name="attention_weights")
    context_dot = Dot(axes=[2, 1], name="context_vector")
    attention_concat = Concatenate(axis=2, name="attention_concat")
    attention_dense = Dense(units, activation='tanh', name="attention_dense")
    output_dense = Dense(ja_vocab_size, activation="softmax", name="final_output")
    
    score = attention_dot1([decoder_lstm_outputs, encoder_outputs_seq])
    weights = attention_softmax(score)
    context = context_dot([weights, encoder_outputs_seq])
    concat = attention_concat([context, decoder_lstm_outputs])
    attentional_vector = attention_dense(concat)
    final_outputs = output_dense(attentional_vector)
    
    model = Model([encoder_inputs, decoder_inputs], final_outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    return model

# --- Translation & Evaluation ---
def translate_sentence(sentence, encoder_model, decoder_model, tokenizer_ja, tokenizer_en, max_len=30):
    sentence_seq = tokenizer_ja.texts_to_sequences(["<s> " + sentence.strip() + " </s>"])
    sentence_seq_np = np.array(sentence_seq)
    
    encoder_outputs_seq, state_h, state_c = encoder_model.predict(sentence_seq_np, verbose=0)
    states = [state_h, state_c]
        
    target_seq = np.array([[tokenizer_en.word_index['<s>']]])
    decoded_sentence = []
    
    for _ in range(max_len):
        decoder_inputs_with_states = [target_seq] + states + [encoder_outputs_seq]
        decoder_outputs = decoder_model.predict(decoder_inputs_with_states, verbose=0)
        output_tokens, new_states = decoder_outputs[0], decoder_outputs[1:]
        
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = tokenizer_en.index_word.get(sampled_token_index, '')
        if sampled_word == '</s>':
            break
        decoded_sentence.append(sampled_word)
        target_seq = np.array([[sampled_token_index]])
        states = new_states
    return ' '.join(decoded_sentence)

# --- Main Execution ---
if __name__ == '__main__':
    # 1. Data Loading & Preprocessing
    print("--- 1. Loading and Preprocessing Data ---")
    ja_sequences, tokenizer_ja = load_data('train.ja')
    en_sequences, tokenizer_en = load_data('train.en')

    if ja_sequences is None or en_sequences is None:
        exit()

    ja_vocab_size = len(tokenizer_ja.word_index) + 1
    en_vocab_size = len(tokenizer_en.word_index) + 1
    
    ja_sequences = pad_sequences(ja_sequences, padding='post')
    en_sequences = pad_sequences(en_sequences, padding='post')

    x_train, x_test, y_train, y_test = train_test_split(en_sequences, ja_sequences, test_size=0.02, random_state=42)

    # 2. Model Training
    print(f"\n--- 2. Building and Training Model (RNN Type: LSTM) ---")
    training_model = create_training_model(ja_vocab_size, en_vocab_size, UNITS)
    training_model.summary()

    train_decoder_input = y_train[:, :-1]
    train_target = y_train[:, 1:]

    training_model.fit([x_train, train_decoder_input], train_target, 
                       batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.2, verbose=1)

    # 3. Saving Artifacts
    print("\n--- 3. Saving Model and Tokenizers ---")
    training_model.save_weights(MODEL_WEIGHTS_PATH)
    with open(TOKENIZER_JA_PATH, 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_ja.to_json(), ensure_ascii=False))
    with open(TOKENIZER_EN_PATH, 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_en.to_json(), ensure_ascii=False))
    print(f"Saved weights to {MODEL_WEIGHTS_PATH}")
    print(f"Saved tokenizers to {TOKENIZER_JA_PATH} and {TOKENIZER_EN_PATH}")

    # 4. Reconstructing Inference Models
    print("\n--- 4. Reconstructing Inference Models from Trained Layers ---")
    
    # Encoder
    encoder_inputs_inf = training_model.inputs[0]
    encoder_outputs_inf = training_model.get_layer('encoder_lstm').output
    encoder_model_inf = Model(encoder_inputs_inf, [encoder_outputs_inf[0]] + encoder_outputs_inf[1:])

    # Decoder
    decoder_embedding_layer = training_model.get_layer('decoder_emb')
    decoder_rnn_layer = training_model.get_layer('decoder_lstm')
    attention_dot1 = training_model.get_layer('attention_score')
    attention_softmax = training_model.get_layer('attention_weights')
    context_dot = training_model.get_layer('context_vector')
    attention_concat = training_model.get_layer('attention_concat')
    attention_dense = training_model.get_layer('attention_dense')
    output_dense = training_model.get_layer('final_output')

    decoder_inputs_inf = Input(shape=(1,), name="decoder_inputs_inf")
    encoder_outputs_seq_inf = Input(shape=(None, UNITS), name="encoder_outputs_seq_inf")
    
    decoder_initial_state_h = Input(shape=(UNITS,), name="decoder_initial_state_h_inf")
    decoder_initial_state_c = Input(shape=(UNITS,), name="decoder_initial_state_c_inf")
    decoder_initial_states_inf = [decoder_initial_state_h, decoder_initial_state_c]

    dec_emb_output_inf = decoder_embedding_layer(decoder_inputs_inf)
    decoder_rnn_outputs_inf, new_state_h, new_state_c = decoder_rnn_layer(dec_emb_output_inf, initial_state=decoder_initial_states_inf)
    decoder_states_inf = [new_state_h, new_state_c]
        
    score = attention_dot1([decoder_rnn_outputs_inf, encoder_outputs_seq_inf])
    weights = attention_softmax(score)
    context = context_dot([weights, encoder_outputs_seq_inf])
    concat = attention_concat([context, decoder_rnn_outputs_inf])
    attentional_vector = attention_dense(concat)
    final_outputs = output_dense(attentional_vector)

    decoder_model_inf = Model(
        [decoder_inputs_inf] + decoder_initial_states_inf + [encoder_outputs_seq_inf],
        [final_outputs] + decoder_states_inf
    )
    print("Inference models reconstructed successfully.")

    # 5. Evaluation
    print(f"\n--- 5. Evaluating on 100 Test Samples ---")
    with open(TOKENIZER_JA_PATH, 'r', encoding='utf-8') as f:
        tokenizer_ja_inf = tokenizer_from_json(json.load(f))
    with open(TOKENIZER_EN_PATH, 'r', encoding='utf-8') as f:
        tokenizer_en_inf = tokenizer_from_json(json.load(f))
    
    test_ja_texts = tokenizer_ja.sequences_to_texts([s for s in y_test[:100]])
    test_en_texts = tokenizer_en.sequences_to_texts([s for s in x_test[:100]])
    test_ja_texts = [re.sub(r'<s>|</s>|<pad>', '', s).strip() for s in test_ja_texts]
    test_en_texts = [re.sub(r'<s>|</s>|<pad>', '', s).strip() for s in test_en_texts]
    
    bleu_scores = []
    chencherry = SmoothingFunction()
    for i, (ja_sent, en_ref) in enumerate(zip(test_ja_texts, test_en_texts)):
        prediction = translate_sentence(ja_sent, encoder_model_inf, decoder_model_inf, tokenizer_ja_inf, tokenizer_en_inf, max_len=MAX_OUTPUT_LENGTH)
        reference_tokens = [en_ref.split()]
        prediction_tokens = prediction.split()
        score = sentence_bleu(reference_tokens, prediction_tokens, smoothing_function=chencherry.method1)
        bleu_scores.append(score)
        if i < 5:
            print(f"\n--- Sentence {i+1} ---")
            print(f"Original (JA): {ja_sent}")
            print(f"Reference (EN): {en_ref}")
            print(f"Prediction (EN): {prediction}")
            print(f"BLEU Score: {score:.4f}")

    avg_bleu = np.mean(bleu_scores) * 100
    print(f"\n=====================================")
    print(f"Average BLEU Score for LSTM with Attention (100 samples): {avg_bleu:.2f}")
    print(f"=====================================")