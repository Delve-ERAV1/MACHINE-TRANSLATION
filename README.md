
# Encoder-Decoder Transformer for English to French Translation

## Overview

This documentation provides a guideline for implementing an encoder-decoder transformer for translating English to French. The transformer will be trained on the "en-fr" dataset from `opus_books`. We will be incorporating various techniques to improve the model's performance and efficiency. The user will manually enter information related to the training and results.

## Dataset Preparation

1. **Source**: Use the "en-fr" dataset from `opus_books`.
2. **Preprocessing**:
   - Remove all English sentences with more than 150 tokens.
   - Remove all French sentences where the length of the French sentence is greater than the length of the corresponding English sentence by more than 10 tokens.

## Model Architecture

### Encoder-Decoder Transformer

The core of our translation task will be the Encoder-Decoder Transformer. This architecture has proven to be highly effective for sequence-to-sequence tasks, especially translation.

```
  | Name    | Type             | Params
---------------------------------------------
0 | model   | Transformer      | 49.8 M
1 | loss_fn | CrossEntropyLoss | 0     
---------------------------------------------
49.8 M    Trainable params
0         Non-trainable params
49.8 M    Total params
199.130   Total estimated model params size (MB)
```

## Techniques Incorporated

### 1. Parameter Sharing

Parameter sharing involves sharing parameters between the encoder and decoder or among the multiple layers of the transformer. It can reduce the number of parameters, making the model lighter and potentially improving generalization.

![image](https://github.com/Delve-ERAV1/S15/assets/11761529/1ecb334d-f17d-4aea-9d8c-5809a0507142)

- **Benefits**:
  - Reduces memory consumption.
  - Can lead to faster convergence.
  - Potentially improves model generalization.
 
  ```python
      # Parameter Sharing CYCLE(REV)
    param_share_idx = list(range(new_N)) + list(range(new_N-1, -1, -1))
    encoder_blocks_pm = [encoder_blocks[idx] for idx in param_share_idx]
    decoder_blocks_pm = [decoder_blocks[idx] for idx in param_share_idx]
  ```

### 2. Multi Query Attention

Instead of having a single set of queries in the attention mechanism, multi-query attention uses multiple sets. This can help the model capture a richer set of relationships in the data.
![Multi-Query Attention is All You Need | by Fireworks.ai | Jul, 2023 | Medium](https://miro.medium.com/v2/resize:fit:1400/0*-ygFb8mX-ctD-z_C)

- **Benefits**:
  - Captures diverse relationships in data.
  - Can lead to improved model performance on complex tasks.

### 3. Gradient Accumulation

Instead of updating the model weights after every mini-batch, gradients are accumulated over multiple mini-batches and then used to update the model. This can be especially useful when working with limited GPU memory, allowing for effectively larger batch sizes.
![image](https://github.com/Delve-ERAV1/S15/assets/11761529/934214d6-ed2f-4d91-8c52-a4da0dadf8a8)

```python
trainer = Trainer(accumulate_grad_batches=5)
```

- **Benefits**:
  - Allows for larger effective batch sizes.
  - Useful for training on limited GPU memory.

### 4. One Cycle Policy

This is a learning rate scheduling technique. The learning rate starts from a low value, increases gradually to a maximum, and then decreases. This policy can help in faster convergence and better performance.

![image](https://github.com/Delve-ERAV1/S15/assets/11761529/72650458-8ca1-45f9-adf5-7d0a18f0997c)

- **Benefits**:
  - Faster convergence.
  - Can prevent overshooting during training.
  - Helps in achieving better performance.

##### LR Finder
![image](https://github.com/Delve-ERAV1/S15/assets/11761529/6dacd615-4477-4ce5-8c2d-d16f96832a6a)


### 5. Dynamic Padding

Instead of padding all sequences in a batch to the maximum sequence length, dynamic padding pads sequences to the maximum length within that batch. This can lead to faster training times as the model processes fewer padded tokens.

![image](https://github.com/Delve-ERAV1/S15/assets/11761529/3e06c1ee-103e-44ff-80a8-2c8073c28bd7)

- **Benefits**:
  - Reduces unnecessary computations.
  - Leads to faster training times.

## Training

```
Epoch 39: 100%|██████████| 848/848 [01:24<00:00, 10.03it/s, v_num=3, train_loss_step=1.680, train_loss_epoch=1.670, validation_cer=0.132, validation_wer=0.343]
```

Training logs may be found [here](training_logs.txt)



## Results

```
=============================================================
SOURCE - Fear of seeing this unheard-of happiness, to which he clung so closely, soon vanish from between his hands?
TARGET - Peur de voir s’évanouir bientôt entre ses mains ce bonheur inouï qu’il tenait si serré ?
PREDICTED - Peur de voir s ’ évanouir bientôt entre ses mains ce bonheur inouï qu ’ il tenait si serré ?
=============================================================
SOURCE - It is to the profits that he has made from his great nail factory that the Mayor of Verrieres is indebted for this fine freestone house which he has just finished building.
TARGET - C’est aux bénéfices qu’il a faits sur sa grande fabrique de clous que le maire de Verrières doit cette belle habitation en pierres de taille qu’il achève en ce moment.
PREDICTED - C ’ est aux bénéfices qu ’ il a fait à sa grande fabrique de clous que le maire de Verrières est cet édifice de belle habitation qu ’ il vient d ’ avoir chaud .
=============================================================
SOURCE - I was careful not to refer to Marguerite, fearing lest the name should awaken sad recollections hidden under the apparent calm of the invalid; but Armand, on the contrary, seemed to delight in speaking of her, not as formerly, with tears in his eyes, but with a sweet smile which reassured me as to the state of his mind.
TARGET - Je me gardais bien de l'entretenir de Marguerite, craignant toujours que ce nom ne réveillât un triste souvenir endormi sous le calme apparent du malade; mais Armand, au contraire, semblait prendre plaisir à parler d'elle, non plus comme autrefois, avec des larmes dans les yeux, mais avec un doux sourire qui me rassurait sur l'état de son âme.
PREDICTED - Je me gardais bien de l ' entretenir de Marguerite , craignant toujours que ce nom ne un triste souvenir endormi sous le calme apparent du malade ; mais Armand , au contraire , semblait prendre plaisir à parler d ' elle , non plus comme autrefois , avec des larmes dans les yeux , mais avec un doux sourire qui me rassurait sur l ' état de son âme .
=============================================================
SOURCE - "And so ended his affection," said Elizabeth impatiently.
TARGET - – Et ainsi, dit Elizabeth avec un peu d’impatience, se termina cette grande passion.
PREDICTED - – Et ainsi , dit Elizabeth avec un peu d ’ impatience , se termina cette grande passion .
=============================================================
```

Prediction logs may be found [here](predictions.log)



