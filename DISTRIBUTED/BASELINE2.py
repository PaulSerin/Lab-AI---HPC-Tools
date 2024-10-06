                        ##################################################
                        #                                                #
                        #               IMPORT LIBRARIES                 #
                        #                                                #
                        ##################################################



import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
import warnings
from torch.utils.tensorboard import SummaryWriter
import datetime


                        ##################################################
                        #                                                #
                        #               PREPROCESS DATA                  #
                        #                                                #
                        ##################################################
                        

pathTrainData = "../data/train-v2.0.json"
pathTestData =  "../data/dev-v2.0.json"

def load_data(file_path):
    """Charge et retourne les contextes, questions et réponses depuis un fichier JSON."""
    with open(file_path, "r") as f:
        data = json.load(f)

    contexts = []
    questions = []
    answers = []

    for group in data['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                for answer in qa['answers']:
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)
    
    return contexts, questions, answers

train_contexts, train_questions, train_answers = load_data(pathTrainData)
test_contexts, test_questions, test_answers = load_data(pathTestData)


def adjust_answer_indices(answers, contexts):
    """
    Adjust the start and end indices of answers to ensure they correctly align
    with the context text. It handles cases where the actual answer might differ
    by one or two characters from the indexed position.
    
    Parameters:
    answers (list): List of answer dictionaries containing 'text' and 'answer_start'.
    contexts (list): List of context strings from which the answers are extracted.
    """
    for answer, context in zip(answers, contexts):
        real_answer = answer['text']
        start_idx = answer['answer_start']
        end_idx = start_idx + len(real_answer)  # Calculate the end index

        # Check if the real answer matches the exact indexed position
        if context[start_idx:end_idx] == real_answer:
            answer['answer_end'] = end_idx
        # Handle case where the real answer is off by one character
        elif context[start_idx-1:end_idx-1] == real_answer:
            answer['answer_start'] = start_idx - 1
            answer['answer_end'] = end_idx - 1
        # Handle case where the real answer is off by two characters
        elif context[start_idx-2:end_idx-2] == real_answer:
            answer['answer_start'] = start_idx - 2
            answer['answer_end'] = end_idx - 2

# Adjust the indices for both training and test sets
adjust_answer_indices(train_answers, train_contexts)
adjust_answer_indices(test_answers, test_contexts)


from transformers import AutoTokenizer,AdamW,BertForQuestionAnswering
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
warnings.filterwarnings("ignore")

train_encodings = tokenizer(
    train_contexts, 
    train_questions, 
    truncation=True, 
    padding=True, 
    # clean_up_tokenization_spaces=True  # Delete warning
)

test_encodings = tokenizer(
    test_contexts, 
    test_questions, 
    truncation=True, 
    padding=True, 
    # clean_up_tokenization_spaces=True 
)


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



                        ##################################################
                        #                                                #
                        #               USE OF DATA LOADER               #
                        #                                                #
                        ##################################################
                        


train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)

small_train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=16,
    sampler=torch.utils.data.SubsetRandomSampler(range(int(0.5 * len(train_dataset))))
)

small_test_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=16,
    sampler=torch.utils.data.SubsetRandomSampler(range(int(0.5 * len(test_dataset))))
)


                        ##################################################
                        #                                                #
                        #                     USE GPU                    #
                        #                                                #
                        ##################################################



# device = torch.device('cuda:0' if torch.cuda.is_available()
#                       else 'cpu')

# if device.type == 'cuda':
#     print(f"Using GPU: {torch.cuda.get_device_name(device)}")
#     print(f"GPU Device Index: {torch.cuda.current_device()}")
#     print(f"Number of GPUs: {torch.cuda.device_count()}")
# else:
#     print("Using CPU")
    
    

                        ##################################################
                        #                                                #
                        #               MODEL AND OPTIMIZER              #
                        #                                                #
                        ##################################################
                        
    
# model = BertForQuestionAnswering.from_pretrained('bert-base-uncased').to(device)

# # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# optim = AdamW(model.parameters(), lr=5e-5)
# # optim = AdamW(model.parameters(), lr=3e-5)
# # optim = AdamW(model.parameters(), lr=2e-5)


                        ##################################################
                        #                                                #
                        #                 TRAINING LOOP                  #
                        #                                                #
                        ##################################################


def train_model(model, train_loader, optimizer, device, epochs=1, print_every=100):
    """
    Train the model for a given number of epochs and print the loss during the training process.
    
    Parameters:
    - model: The model to be trained.
    - train_loader: The DataLoader for training data.
    - optimizer: Optimizer used for training.
    - device: The device (CPU/GPU) on which to train the model.
    - epochs: Number of epochs to train the model (default=1).
    - print_every: Interval of batches to print the training status (default=100).
    
    Returns:
    - train_losses: List of average losses per epoch.
    """
    model.train()
    train_losses = []
    
    total_train_time = time.time()  # Start timing the entire training process

    for epoch in range(epochs):
        epoch_time = time.time()
        total_loss = 0
        
        print(f"\n############ Train Epoch {epoch + 1} ############")
        for batch_idx, batch in enumerate(tqdm(train_loader, disable=True)):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
            loss = outputs[0]
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            writer.add_scalar('Training Loss', loss.item(), epoch * len(train_loader) + batch_idx)

            if (batch_idx + 1) % print_every == 0:
                print(f"Batch {batch_idx + 1} / {len(train_loader)}, Loss: {round(loss.item(), 1)}")
        
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch + 1} finished. Training Loss: {avg_loss}, Time: {time.time() - epoch_time}\n")

    total_train_time = time.time() - total_train_time 
    minutes = total_train_time // 60
    seconds = total_train_time % 60
    print(f"Total Training Time: {int(minutes)} minutes {seconds:.2f} seconds")

    return train_losses


                        ##################################################
                        #                                                #
                        #                EVALUATION LOOP                 #
                        #                                                #
                        ##################################################


# def evaluate_model(model, eval_loader, device, print_every=100):
#     """
#     Evaluate the model on the evaluation dataset.
    
#     Parameters:
#     - model: The model to be evaluated.
#     - eval_loader: The DataLoader for evaluation data.
#     - device: The device (CPU/GPU) on which to evaluate the model.
#     - print_every: Interval of batches to print the evaluation status (default=1000).
    
#     Returns:
#     - eval_losses: List of average losses per evaluation.
#     """
#     model.eval()
#     eval_losses = []
#     total_loss = 0
    
#     total_train_time = time.time()  # Start timing the entire training process
    
#     print("\n############ Evaluation ############")
#     with torch.no_grad():
#         for batch_idx, batch in enumerate(eval_loader):
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             start_positions = batch['start_positions'].to(device)
#             end_positions = batch['end_positions'].to(device)

#             outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
#             loss = outputs[0]
            
#             total_loss += loss.item()
            
#             writer.add_scalar('Evaluation Loss', loss.item(), batch_idx)

#             if (batch_idx + 1) % print_every == 0:
#                 print(f"Batch {batch_idx + 1} / {len(eval_loader)}, Loss: {round(loss.item(), 1)}")
        
#     avg_loss = total_loss / len(eval_loader)
#     eval_losses.append(avg_loss)
#     print(f"Evaluation finished. Validation Loss: {avg_loss}\n")
    
#     total_train_time = time.time() - total_train_time 
#     minutes = total_train_time // 60
#     seconds = total_train_time % 60
#     print(f"Total Evaluation Time: {int(minutes)} minutes {seconds:.2f} seconds")
    
#     return eval_losses

# def train_and_evaluate(model, train_loader, eval_loader, optimizer, device, epochs=1, print_every=100):
#     """
#     Train and evaluate the model.
    
#     Parameters:
#     - model: The model to be trained and evaluated.
#     - train_loader: The DataLoader for training data.
#     - eval_loader: The DataLoader for evaluation data.
#     - optimizer: Optimizer used for training.
#     - device: The device (CPU/GPU) on which to train and evaluate the model.
#     - epochs: Number of epochs to train the model.
#     - print_every: Interval of batches to print the status.
    
#     Returns:
#     - train_losses: List of training losses per epoch.
#     - eval_losses: List of evaluation losses per epoch.
#     """
#     whole_train_eval_time = time.time()

#     train_losses = train_model(model, train_loader, optimizer, device, epochs, print_every)
#     eval_losses = evaluate_model(model, eval_loader, device, print_every)
    
#     total_train_time = time.time() - total_train_time 
#     minutes = total_train_time // 60
#     seconds = total_train_time % 60
#     print(f"Total Training and Evaluation Time: {int(minutes)} minutes {seconds:.2f} seconds")
    
#     return train_losses, eval_losses

import pytorch_lightning as pl


# Adaptation du modèle en LightningModule
class BertLightning(pl.LightningModule):
    def __init__(self, model_name='bert-base-uncased', learning_rate=5e-5):
        super().__init__()
        self.model = BertForQuestionAnswering.from_pretrained(model_name)
        self.learning_rate = learning_rate

    def forward(self, input_ids, attention_mask, start_positions, end_positions):
        return self.model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        start_positions = batch['start_positions']
        end_positions = batch['end_positions']

        outputs = self(input_ids, attention_mask, start_positions, end_positions)
        loss = outputs[0]

        # Logging training loss
        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        start_positions = batch['start_positions']
        end_positions = batch['end_positions']

        outputs = self(input_ids, attention_mask, start_positions, end_positions)
        loss = outputs[0]

        # Logging validation loss
        self.log('val_loss', loss)

    def configure_optimizers(self):
        return AdamW(self.model.parameters(), lr=self.learning_rate)



                        ##################################################
                        #                                                #
                        #              LUANCHING FONCTIONS               #
                        #                                                #
                        ##################################################



print("""
                        ##################################################
                        #                                                #
                        #         START TRAINING AND EVALUATION          #
                        #                                                #
                        ##################################################
""")


# log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# writer = SummaryWriter(log_dir)

# train_losses = train_model(model, train_loader, optim, device, epochs=2)
# eval_losses = evaluate_model(model, test_loader, device, print_every=100,)

# writer.close()


# Méthode 1: DDP (Distributed Data Parallel)
model_ddp = BertLightning()
trainer_ddp = pl.Trainer(accelerator="gpu", devices=2, precision=16, max_epochs=2, strategy="ddp")

start_time_ddp = time.time()  # Démarre le chronomètre
trainer_ddp.fit(model_ddp, small_train_loader, small_test_loader)
end_time_ddp = time.time()    # Arrête le chronomètre

time_ddp = end_time_ddp - start_time_ddp  # Calcule le temps écoulé
print(f"Temps d'entraînement avec DDP: {time_ddp:.2f} secondes")

# Méthode 2: FSDP (Fully Sharded Data Parallel)
model_fsdp = BertLightning()
trainer_fsdp = pl.Trainer(accelerator="gpu", devices=2, precision=16, max_epochs=2, strategy="fsdp")

start_time_fsdp = time.time()  # Démarre le chronomètre
trainer_fsdp.fit(model_fsdp, small_train_loader, small_test_loader)
end_time_fsdp = time.time()    # Arrête le chronomètre

time_fsdp = end_time_fsdp - start_time_fsdp  # Calcule le temps écoulé
print(f"Temps d'entraînement avec FSDP: {time_fsdp:.2f} secondes")


print("""
                        ##################################################
                        #                                                #
                        #                      END                       #
                        #                                                #
                        ##################################################
""")
