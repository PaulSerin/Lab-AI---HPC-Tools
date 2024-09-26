
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import time


with open("./train-v2.0.json", "r") as f:
    train_data = json.load(f)

contexts = []
questions = []
answers = []

# Search for each passage, its question and its answer
for group in train_data['data']:
    for passage in group['paragraphs']:
        context = passage['context']
        for qa in passage['qas']:
            question = qa['question']
            for answer in qa['answers']:
                # Store every passage, query and its answer to the lists
                contexts.append(context)
                questions.append(question)
                answers.append(answer)

train_contexts, train_questions, train_answers = contexts, questions, answers

print(train_contexts[0])
print(train_questions[0])
print(train_answers[0])


with open("./dev-v2.0.json", "r") as f:
    test_data = json.load(f)

contexts = []
questions = []
answers = []

# Search for each passage, its question and its answer
for group in test_data['data']:
    for passage in group['paragraphs']:
        context = passage['context']
        for qa in passage['qas']:
            question = qa['question']
            for answer in qa['answers']:
                # Store every passage, query and its answer to the lists
                contexts.append(context)
                questions.append(question)
                answers.append(answer)

test_contexts, test_questions, test_answers = contexts, questions, answers

# print(test_contexts[0])
# print(test_questions[0])
# print(test_answers[0])


# print(len(train_contexts))
# print(len(train_questions))
# print(len(train_answers))

# print("Passage: ",train_contexts[0])
# print("Query: ",train_questions[0])
# print("Answer: ",train_answers[0])


# print(len(test_contexts))
# print(len(test_questions))
# print(len(test_answers))

# print("Passage: ",test_contexts[0])
# print("Query: ",test_questions[0])
# print("Answer: ",test_answers[0])



# train_texts = train_texts[:10]
# train_queries = train_queries[:10]
# train_answers = train_answers[:10]

# testtt_texts ttestttal_texts[:10]
# testt_queries = test_queries[:10]
# test_answers = test_answers[:10]

# print(len(train_texts))
# print(len(train_queries))
# print(len(train_answers))

# print(len(test_contexts))
# print(len(test_queries))
# print(len(test_answers))




for answer, context in zip(train_answers, train_contexts):
    real_answer = answer['text']
    start_idx = answer['answer_start']
    # Get the real end index
    end_idx = start_idx + len(real_answer)

    # Deal with the problem of 1 or 2 more characters
    if context[start_idx:end_idx] == real_answer:
        answer['answer_end'] = end_idx
    # When the real answer is more by one character
    elif context[start_idx-1:end_idx-1] == real_answer:
        answer['answer_start'] = start_idx - 1
        answer['answer_end'] = end_idx - 1
    # When the real answer is more by two characters
    elif context[start_idx-2:end_idx-2] == real_answer:
        answer['answer_start'] = start_idx - 2
        answer['answer_end'] = end_idx - 2



for answer, context in zip(test_answers, test_contexts):
    real_answer = answer['text']
    start_idx = answer['answer_start']
    # Get the real end index
    end_idx = start_idx + len(real_answer)

    # Deal with the problem of 1 or 2 more characters
    if context[start_idx:end_idx] == real_answer:
        answer['answer_end'] = end_idx
    # When the real answer is more by one character
    elif context[start_idx-1:end_idx-1] == real_answer:
        answer['answer_start'] = start_idx - 1
        answer['answer_end'] = end_idx - 1
    # When the real answer is more by two characters
    elif context[start_idx-2:end_idx-2] == real_answer:
        answer['answer_start'] = start_idx - 2
        answer['answer_end'] = end_idx - 2


from transformers import AutoTokenizer,AdamW,BertForQuestionAnswering
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)
test_encodings = tokenizer(test_contexts, test_questions, truncation=True, padding=True)



def add_token_positions(encodings, answers):
  start_positions = []
  end_positions = []

  count = 0

  for i in range(len(answers)):
    start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
    end_positions.append(encodings.char_to_token(i, answers[i]['answer_end']))

    # if start position is None, the answer passage has been truncated
    if start_positions[-1] is None:
      start_positions[-1] = tokenizer.model_max_length

    # if end position is None, the 'char_to_token' function points to the space after the correct token, so add - 1
    if end_positions[-1] is None:
      end_positions[-1] = encodings.char_to_token(i, answers[i]['answer_end'] - 1)
      # if end position is still None the answer passage has been truncated
      if end_positions[-1] is None:
        count += 1
        end_positions[-1] = tokenizer.model_max_length

  print(count)

  # Update the data in dictionary
  encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

add_token_positions(train_encodings, train_answers)
add_token_positions(test_encodings, test_answers)




class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

train_dataset = SquadDataset(train_encodings)
test_dataset = SquadDataset(test_encodings)



train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)

small_train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=16,
    sampler=torch.utils.data.SubsetRandomSampler(range(int(0.3 * len(train_dataset))))
)

"""## ***Step 9:*** Use GPU"""

device = torch.device('cuda:0' if torch.cuda.is_available()
                      else 'cpu')

"""## ***Step 10:*** Build the Bert model

I select BertForQuestionAnswering from transformers library, as it was the most relative with this task. When we instantiate a model with from_pretrained(), the model configuration and pre-trained weights of the specified model are used to initialize the model. Moreover, I used the PyTorch optimizer of AdamW which implements gradient bias correction as well as weight decay.
"""

model = BertForQuestionAnswering.from_pretrained('bert-base-uncased').to(device)

optim = AdamW(model.parameters(), lr=5e-5)
# optim = AdamW(model.parameters(), lr=3e-5)
# optim = AdamW(model.parameters(), lr=2e-5)

# epochs = 2
epochs = 1
# epochs = 4

"""This message is a warning that I should fine tune my model before I test it, in order to have a good performance.

## ***Step 11:*** Train and Evaluate Model

Training of model was done exactly as in the previous projects.
"""

whole_train_eval_time = time.time()

train_losses = []
test_losses = []

print_every = 1000

for epoch in range(epochs):
  epoch_time = time.time()

  # Set model in train mode
  model.train()

  loss_of_epoch = 0

  print("############Train############")

  print("longueur de small_train_loader:", len(small_train_loader))

  for batch_idx,batch in enumerate(small_train_loader):

    print(batch_idx)

    optim.zero_grad()

    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    start_positions = batch['start_positions'].to(device)
    end_positions = batch['end_positions'].to(device)

    outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
    loss = outputs[0]
    # do a backwards pass
    loss.backward()
    # update the weights
    optim.step()
    # Find the total lossll
    loss_of_epoch += loss.item()

    if (batch_idx+1) % print_every == 0:
      print("Batch {:} / {:}".format(batch_idx+1,len(small_train_loader)),"\nLoss:", round(loss.item(),1),"\n")

  loss_of_epoch /= len(train_loader)
  train_losses.append(loss_of_epoch)

  ##########Evaluation##################

  # Set model in etestuation mode
  model.eval()

  print("############Evaluate############")

  loss_of_epoch = 0

  for batch_idx,batch in enumerate(small_train_loader):

    with torch.no_grad():

      input_ids = batch['input_ids'].to(device)
      attention_mask = batch['attention_mask'].to(device)
      start_positions = batch['start_positions'].to(device)
      end_positions = batch['end_positions'].to(device)

      outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
      loss = outputs[0]
      # Find the total loss
      loss_of_epoch += loss.item()

    if (batch_idx+1) % print_every == 0:
       print("Batch {:} / {:}".format(batch_idx+1,len(test_loader)),"\nLoss:", round(loss.item(),1),"\n")

  loss_of_epoch /= len(small_train_loader)
  test_losses.append(loss_of_epoch)

  # Print each epoch's time and train/test loss
  print("\n-------Epoch ", epoch+1,
        "-------"
        "\nTraining Loss:", train_losses[-1],
        "\ntestidation Loss:", test_losses[-1],
        "\nTime: ",(time.time() - epoch_time),
        "\n-----------------------",
        "\n\n")

print("Total training and etestuation time: ", (time.time() - whole_train_eval_time))

from google.colab import drive
drive.mount('/content/drive')
torch.save(model, "/content/drive/MyDrive/finetunedmodel.pth")

"""## ***Step 12:*** Plot train and validation losses"""

import matplotlib.pyplot as plt

fig,ax = plt.subplots(1,1,figsize=(15,10))

ax.set_title("Train and Validation Losses",size=20)
ax.set_ylabel('Loss', fontsize = 20)
ax.set_xlabel('Epochs', fontsize = 25)
_=ax.plot(train_losses)
_=ax.plot(test_losses)
_=ax.legend(('Train','Val'),loc='upper right')