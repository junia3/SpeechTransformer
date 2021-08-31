import torch
import torch.nn as nn
import torch.optim as optim
from speech_transformer.model import SpeechTransformer
from speech_transformer.modules import SubSampling
from torch.utils.data import DataLoader
from dataload import PhonemeDataset
import os
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

########## training data, validation data csv file path ##############
train_path = os.path.join(os.getcwd(), 'librispeech_100h_sequence.csv')
val_path = os.path.join(os.getcwd(), 'librispeech_dev_sequence.csv')

########## arguments for mel filterbank features ###############
args = {}
args['sample_rate'] = 16000
args['n_fft'] = 512
args['win_length'] = 480
args['hop_length'] = 160
args['f_max'] = 8000
args['n_mels'] = 40


############## get data for training & validation ###############
train_data = PhonemeDataset(train_path, os.getcwd(), args = args)
val_data = PhonemeDataset(val_path, os.getcwd(), args = args)

############ collate function for phoneme ######################
def phonecollate(batch):
    max_len = 0
    input_length = []
    output_length = []
    for x, y in batch:
       max_len = max(max_len, x.shape[-1])
       input_length.append(int(x.shape[-1]))
       output_length.append(int(y.shape[-1]))
    ## pad max length for each batch
    new_x = torch.ones(1, x.shape[1], max_len)
    new_y = torch.ones(1, max_len)
    subsample = SubSampling(max_len)
    for x, y in batch:
         temp_x = F.pad(x, [0, max_len - x.shape[-1]])
         temp_y = torch.cat((y,  40 * torch.ones(max_len - y.shape[-1])))
         new_x = torch.cat((new_x, temp_x), 0)
         new_y = torch.cat((new_y, temp_y.unsqueeze(0)), 0)
    new_x = new_x[1:,:,:].transpose(-2, -1)  ## batch * frame * feature
    new_y = new_y[1:,:].type(torch.IntTensor)  ## batch * frame
    
    target_input = torch.cat((43 * torch.ones(new_y.shape[0], 1), new_y[:, :-1]), axis = 1)
    target_output = torch.cat((new_y[:, 1:], 44 * torch.ones(new_y.shape[0], 1)), axis = 1)

    target_input, target_output = target_input.type(torch.IntTensor), target_output.type(torch.LongTensor)
    
    new_x = subsample(new_x)
    return new_x, target_input, target_output, input_length, output_length

############ dataloader ###################
train_dataloader = DataLoader(dataset = train_data, batch_size = 1, shuffle = True, collate_fn = phonecollate)
val_dataloader = DataLoader(dataset = val_data, batch_size = 1, shuffle = True, collate_fn = phonecollate)


############ device ##################
USE_CUDA = torch.cuda.is_available()
print('GPU : ',USE_CUDA)
device = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('Device is : ',device)



#### model ################3
model = SpeechTransformer(d_model = 256, d_ff = 1024).cuda()

#### training model #####
def train_data(ep, train_dataloader, val_dataloader, warm, criterion = 'CE'):
    train_loss = []
    validation_loss = []
    accuracy = []

    optimizer = optim.Adam(model.parameters(), betas = (0.9, 0.98), eps = 1e-9, lr = 8e-4)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda step : 256 ** (-0.5) * min((step + 1) ** (-0.5), (step + 1) * warm ** (-1.5)), verbose = False)

    if criterion == 'CE':
        score_function = nn.CrossEntropyLoss()
    elif criterion == 'CTC':
        score_function = nn.CTCLoss()
    else:
        raise ValueError("Not valid loss function in this model!")
    running_loss = 0.0
    
    for epoch in range(ep):
      for i, data in enumerate(train_dataloader, 0):
          inputs, target_input, target_output, input_length, output_length = data
          input_length = torch.tensor(input_length, dtype = torch.int32)
          output_length = torch.tensor(output_length, dtype = torch.int32)
          optimizer.zero_grad()
          
          #### gpu ####
          inputs, target_input, target_output, input_length, output_length = inputs.cuda(), target_input.cuda(), target_output.cuda(), input_length.cuda(), output_length.cuda()
          
          predictions, logits = model(inputs, input_length, target_input, output_length)
          ### get predictions for phonemes
          
          predictions = predictions.view(-1) ## batch * frame
          logits = logits.reshape(logits.shape[0] * logits.shape[1], logits.shape[2]) ## batch * frame * class
          target_output = target_output.view(-1) ## batch * frame
          
          loss = score_function(logits.cuda(), target_output.cuda())
          loss.backward()
          optimizer.step()
          running_loss += loss.item()
          if i % 10  == 0:
              print('[%d, %5d] loss : %.3f'%(epoch + 1, i, running_loss / 10))
              train_loss.append(running_loss / 10)
              running_loss = 0.0
          
          if i % 200 == 199:
            with torch.no_grad():
              val_loss = 0.0
              acc = 0.0
              for j, data in enumerate(val_dataloader):
                  inputs, target_input, target_output, input_length, output_length = data
                  input_length = torch.tensor(input_length, dtype = torch.int32)
                  output_length = torch.tensor(output_length, dtype = torch.int32)
                  inputs, target_input, target_output, input_length, output_length = inputs.cuda(), target_input.cuda(), target_output.cuda(), input_length.cuda(), output_length.cuda()
                  val_pred, val_logits = model(inputs, input_length, target_input, output_length)
                  
                  val_pred = val_pred.view(-1).type(torch.IntTensor)
                  val_logits = val_logits.reshape(val_logits.shape[0] * val_logits.shape[1], val_logits.shape[2])
                  target_output = target_output.view(-1)
                  
                  val_loss = (val_loss*j + score_function(val_logits.cuda(), target_output.cuda()))/(j+1)

                  acc = (acc*j + torch.sum(val_pred == target_output.cpu()) / target_output.shape[0]) / (j+1)
                  
                  if(j % 200 == 199):
                      break
              validation_loss.append(val_loss)
              accuracy.append(acc * 100)
              print("validation loss : %.3f, accuracy : %.3f%%"%(val_loss, acc * 100) )
          scheduler.step() 
          if i % 2500 == 2499:
              break
    print('finished traning')
    
    return train_loss, validation_loss, accuracy


results = train_data(5, train_dataloader, val_dataloader, 100, 'CE')

train_loss, val_loss, accuracy = results

plt.figure(1)
plt.plot(range(1, len(train_loss)+1), train_loss)
plt.title('Training loss');
plt.savefig('train.png')

plt.figure(2)
plt.plot(range(1, len(val_loss)+1), val_loss)
plt.title('Validation loss');
plt.savefig('validation.png')

plt.figure(3)
plt.plot(range(1, len(accuracy)+1), accuracy)
plt.title('accuracy');
plt.savefig('accuracy.png')
