from tokenizers.models import BPE
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import Sequence, Lowercase
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from transformers import GPT2TokenizerFast, GPT2Config,TFGPT2LMHeadModel
import tensorflow as tf
import numpy as np


tokenizer = Tokenizer(BPE())
tokenizer.normalizer = Sequence([
    Lowercase()
])
tokenizer.pre_tokenizer = ByteLevel()
tokenizer.decoder = ByteLevelDecoder()

trainer = BpeTrainer(vocab_size=50000, initial_alphabet=ByteLevel.alphabet(),special_tokens=[
    "<a>","<pad>","</s>","<unk>","<mask>"
])
tokenizer.train(["/Users/chris/Documents/Coffee_Language_Processor/James_Clean.txt"],trainer)

tokenizer.save("/Users/chris/Documents/Coffee_Language_Processor/tokenizer_gpt/tokenizer.json")

tokenizer_gpt = GPT2TokenizerFast.from_pretrained("tokenizer_gpt")

tokenizer_gpt.add_special_tokens({
    "eos_token":"</s>",
    "bos_token": "<s>",
    "unk_token": "<unk>",
    "pad_token": "<pad>",
    "mask_token": "<mask>"
})

def generate(start, model):
    input_token_ids = tokenizer_gpt.encode(start, return_tensors='tf')
    output = model.generate(input_token_ids,max_length = 500,
            num_beams = 5,
            temperature = 0.7,
            no_repeat_ngram_size=2,
            num_return_sequences=1
            )
    return tokenizer_gpt.decode(output[0])



# Recreate the exact same model, including its weights and the optimizer
new_model = tf.keras.models.load_model('/Users/chris/Documents/Coffee_Language_Processor/Hoffman_GPT_Simple_5')

# Show the model architecture
new_model.summary()

config = GPT2Config(
    vocab_size=tokenizer_gpt.vocab_size,
    bos_token_id=tokenizer_gpt.bos_token_id,
    eos_token_id=tokenizer_gpt.eos_token_id
)

optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5,epsilon=1e-08, clipnorm=1.0)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
new_model.compile(optimizer=optimizer, loss=[loss, *[None] * new_model.config.n_layer], metrics=[metric])

print(generate("espresso", new_model)) 
