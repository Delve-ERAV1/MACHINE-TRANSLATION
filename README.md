
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
0 | model   | Transformer      | 53.5 M
1 | loss_fn | CrossEntropyLoss | 0     
---------------------------------------------
53.5 M    Trainable params
0         Non-trainable params
53.5 M    Total params
213.821   Total estimated model params size (MB)
```

## Techniques Incorporated

### 1. Parameter Sharing

Parameter sharing involves sharing parameters between the encoder and decoder or among the multiple layers of the transformer. It can reduce the number of parameters, making the model lighter and potentially improving generalization.

![image](https://github.com/Delve-ERAV1/S15/assets/11761529/1ecb334d-f17d-4aea-9d8c-5809a0507142)

- **Benefits**:
  - Reduces memory consumption.
  - Can lead to faster convergence.
  - Potentially improves model generalization.

### 2. Multi Query Attention

Instead of having a single set of queries in the attention mechanism, multi-query attention uses multiple sets. This can help the model capture a richer set of relationships in the data.
![Multi-Query Attention is All You Need | by Fireworks.ai | Jul, 2023 | Medium](https://miro.medium.com/v2/resize:fit:1400/0*-ygFb8mX-ctD-z_C)

- **Benefits**:
  - Captures diverse relationships in data.
  - Can lead to improved model performance on complex tasks.

### 3. Gradient Accumulation

Instead of updating the model weights after every mini-batch, gradients are accumulated over multiple mini-batches and then used to update the model. This can be especially useful when working with limited GPU memory, allowing for effectively larger batch sizes.
![image](https://github.com/Delve-ERAV1/S15/assets/11761529/934214d6-ed2f-4d91-8c52-a4da0dadf8a8)

- **Benefits**:
  - Allows for larger effective batch sizes.
  - Useful for training on limited GPU memory.

### 4. One Cycle Policy

This is a learning rate scheduling technique. The learning rate starts from a low value, increases gradually to a maximum, and then decreases. This policy can help in faster convergence and better performance.

![image](https://github.com/Delve-ERAV1/S15/assets/11761529/6dacd615-4477-4ce5-8c2d-d16f96832a6a)


![image](https://github.com/Delve-ERAV1/S15/assets/11761529/72650458-8ca1-45f9-adf5-7d0a18f0997c)

- **Benefits**:
  - Faster convergence.
  - Can prevent overshooting during training.
  - Helps in achieving better performance.

### 5. Dynamic Padding

Instead of padding all sequences in a batch to the maximum sequence length, dynamic padding pads sequences to the maximum length within that batch. This can lead to faster training times as the model processes fewer padded tokens.

![image](https://github.com/Delve-ERAV1/S15/assets/11761529/3e06c1ee-103e-44ff-80a8-2c8073c28bd7)

- **Benefits**:
  - Reduces unnecessary computations.
  - Leads to faster training times.

## Training

```
Epoch 28:  44%|████▎     | 169/388 [00:26<00:34,  6.40it/s, v_num=14, train_loss_step=1.730, train_loss_epoch=1.780, validation_cer=0.043, validation_wer=0.421]
```

Training logs may be found [here](training_logs.txt)



## Results

```
=============================================================

SOURCE - The scholar finally reached the balcony of the gallery, and climbed over it nimbly, to the applause of the whole vagabond tribe.

TARGET - L’écolier toucha enfin au balcon de la galerie, et l’enjamba assez lestement aux applaudissements de toute la truanderie.

PREDICTED - L ’ écolier toucha enfin au balcon de la galerie , et l ’ on la touchait presque tout entière à la tribu des applaudissements .

=============================================================

SOURCE - The weather was magnificent.

TARGET - Le temps était magnifique.

PREDICTED - Le temps était magnifique .

=============================================================

SOURCE - Despite the distance, despite the noise of wind and sea, we could distinctly hear the fearsome thrashings of the animal's tail, and even its panting breath.

TARGET - Malgré la distance, malgré le bruit du vent et de la mer, on entendait distinctement les formidables battements de queue de l'animal et jusqu'à sa respiration haletante.

PREDICTED - Malgré la distance , malgré le bruit du vent et de la mer , on entendait distinctement les formidables battements de queue de l ' animal et jusqu ' à sa respiration haletante .

=============================================================

SOURCE - "Then, besides that," continued Prudence; "admit that Marguerite loves you enough to give up the count or the duke, in case one of them were to discover your liaison and to tell her to choose between him and you, the sacrifice that she would make for you would be enormous, you can not deny it.

TARGET - Puis, outre cela, admettons, continua Prudence, que Marguerite vous aime assez pour renoncer au comte et au duc, dans le cas où celui-ci s'apercervrait de votre liaison et lui dirait de choisir entre vous et lui, le sacrifice qu'elle vous ferait serait énorme, c'est incontestable.

PREDICTED - -- Alors , ce qui vous aime , reprit Prudence , que Marguerite vous aime assez de renoncer au comte ou d ' un de la permission d ' entre vous et vous les lui direz à celle - ci , et vous dans le cas où elle ne pas au comte , vous !

=============================================================

```

Prediction logs may be found [here](predictions.log)



