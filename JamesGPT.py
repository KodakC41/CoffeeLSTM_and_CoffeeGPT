# GPT 

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

config = GPT2Config(
    vocab_size=tokenizer_gpt.vocab_size,
    bos_token_id=tokenizer_gpt.bos_token_id,
    eos_token_id=tokenizer_gpt.eos_token_id
)
model = TFGPT2LMHeadModel(config)

with open("James_Clean_Simple.txt", "r", encoding='utf-8') as f:
    content = f.readlines()

content_p = []
for c in content:
    if len(c)>10:
        content_p.append(c.strip())
content_p = " ".join(content_p)+tokenizer_gpt.eos_token

tokenized_content = tokenizer_gpt.encode(content_p)

sample_len = 100
examples = []
for i in range(0,len(tokenized_content)):
    examples.append(tokenized_content[i:i + sample_len])

print(examples[1])

train_data = []
labels = []
for example in examples:
    if len(example) == 100:
        train_data.append(example[:-1])
        labels.append(example[1:])

train_data = np.array(train_data).astype(np.int32)
labels = np.array(labels).astype(np.int32)

buffer = 200
batch_size = 30
dataset = tf.data.Dataset.from_tensor_slices((train_data,labels))
dataset = dataset.shuffle(buffer).batch(batch_size,drop_remainder=True)

optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5,epsilon=1e-08, clipnorm=1.0)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
model.compile(optimizer=optimizer, loss=[loss, *[None] * model.config.n_layer], metrics=[metric])

epochs = 5
model.fit(dataset, epochs=epochs)

def generate(start, model):
    input_token_ids = tokenizer_gpt.encode(start, return_tensors='tf')
    output = model.generate(input_token_ids,max_length = 500,
            num_beams = 5,
            temperature = 0.7,
            no_repeat_ngram_size=2,
            num_return_sequences=1
            )
    return tokenizer_gpt.decode(output[0])
model.save_weights("fine_tuned_gpt.h5")
model.save("Hoffman_GPT_Simple_{epochs}".format(epochs = epochs))
print(generate("espresso", model)) 

