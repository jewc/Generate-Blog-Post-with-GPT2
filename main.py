# pip install transformers

import tensorflow as tf
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# create variable tokenizer, load up pre-trained gpt2-large
# leverage GPT2 large model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")

# token to pad text
# end of sentence token id
model = GPT2LMHeadModel.from_pretrained("gpt2-large", pad_token_id=tokenizer.eos_token_id)

# Set-up a sentence and encode it
# Tokenization is the process of converting a string into a sequence of numbers
# Tokens can then be passed into GPT2
sentence = 'I like to eat sushi and I cannot lie'
input_ids = tokenizer.encode(sentence, return_tensors='pt')
input_ids

# generate text until the output length (which includes the context length) reaches 100
# beam search limited to 5
# no_repeat_ngram_size=2 ,stops model from repeating words over and over again
output = model.generate(input_ids, max_length=100, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
output

# decode
output_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(output_text)

# with open('blogpost.txt', 'w') as f:
#    f.write(output_text)
#
