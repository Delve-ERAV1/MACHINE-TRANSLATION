import torch
import logging
import warnings
import torchmetrics
from torch import nn
from config import get_config
import lightning.pytorch as pl
from train import get_ds, get_model
from torch.nn import functional as F
from torch.optim.lr_scheduler import OneCycleLR
from lightning.pytorch.loggers import TensorBoardLogger
from torchmetrics.text import CharErrorRate, WordErrorRate
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor


logger = logging.getLogger("Transformer")
logger.setLevel(level=logging.INFO)
fileHandler = logging.FileHandler(filename='prediction.log')
fileHandler.setLevel(level=logging.INFO)
logger.addHandler(fileHandler)

class TransformerLightning(pl.LightningModule):
  def __init__(self, config, tokenizer_src, tokenizer_tgt, tr_len, label_smoothing=0.1):
    super().__init__()
    self.tr_len = tr_len
    self.expected = []
    self.predicted = []
    self.initial_epoch = 0
    self.config = config
    self.tokenizer_src = tokenizer_src
    self.tokenizer_tgt = tokenizer_tgt
    self.model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size())
    self.loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=label_smoothing)
    self.save_hyperparameters()

  def forward(self, x):
    return self.model(x)
  
  def configure_optimizers(self):
    optimizer = torch.optim.AdamW(self.parameters(), 
                                 lr=self.config['lr'], 
                                 eps=1e-9)
    scheduler = OneCycleLR(
      optimizer,
      max_lr=self.config['lr'],
      pct_start=0.5,
      epochs=self.trainer.max_epochs,
      steps_per_epoch=self.num_steps(),
      anneal_strategy='linear',
      div_factor=100,
      final_div_factor=100,
      three_phase=True
    )
    return {'optimizer': optimizer, 
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}


  def training_step(self, batch, batch_idx):
    encoder_input = batch['encoder_input'] # (b, seq_len)
    decoder_input = batch['decoder_input'] # (b, seq_len)
    encoder_mask = batch['encoder_mask'] # (b, 1, 1, seq_len)
    decoder_mask = batch['decoder_mask'] # (b, 1, seq_len, seq_len)

    # Run the tensors through the encoder, decoder and the projection layer
    encoder_output = self.model.encode(encoder_input, encoder_mask) # (b, seq_len, d_model)
    decoder_output = self.model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
    proj_output = self.model.project(decoder_output) # ( b, seq_len, vocab_size)

    # Compare the output with the label
    label = batch['label'] # (b, seq_len)

    # Compute the loss using cross entropy
    loss = self.loss_fn(proj_output.view(-1, self.tokenizer_tgt.get_vocab_size()), label.view(-1))
    self.log('train_loss', loss.item(), prog_bar=True, on_epoch=True, on_step=True, logger=True)
    return loss


  def validation_step(self, batch, batch_idx):
    encoder_input = batch['encoder_input'] # (b, seq_len)
    encoder_mask = batch['encoder_mask'] # (b, 1, 1, seq_len)

    assert encoder_input.size(
                0
            ) == 1, "Batch Size must be 1 for validation"
    
    model_out = self.greedy_decode(encoder_input, encoder_mask)

    source_text = batch["src_text"][0]
    target_text = batch["tgt_text"][0]
    model_out_text = self.tokenizer_tgt.decode(model_out.detach().cpu().numpy())

    logger.info(f"SOURCE - {source_text}")
    logger.info(f"TARGET - {target_text}")
    logger.info(f"PREDICTED - {model_out_text}")
    logger.info("=============================================================")

    self.expected.append(target_text)
    self.predicted.append(model_out_text)

  def on_validation_epoch_end(self):
    metric = CharErrorRate()
    cer = metric(self.predicted, self.expected)
    self.log('validation_cer', cer, prog_bar=True, on_epoch=True, logger=True)


    # Compute the word error rate
    metric = WordErrorRate()
    wer = metric(self.predicted, self.expected)
    self.log('validation_wer', wer, prog_bar=True, on_epoch=True, logger=True)

    self.expected.clear()
    self.predicted.clear()

  def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
    optimizer.zero_grad(set_to_none=True)

  def greedy_decode(self, source, source_mask):
    sos_idx = self.tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = self.tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = self.model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source)

    while True:
        if decoder_input.size(1) == self.config['seq_len']:
            break

        # Build mask for target
        decoder_mask = self.causal_mask(decoder_input.size(1)).type_as(source_mask)

        # Calculate the output
        out = self.model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # Get next token
        prob = self.model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item())], 
            dim=1
        )

        if next_word == eos_idx:
            break

    return(decoder_input.squeeze(0))


  def causal_mask(self, size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return(mask == 0)

  def num_steps(self) -> int:
      """Get number of steps"""
      # Accessing _data_source is flaky and might break
      dataset = self.trainer.fit_loop._data_source.dataloader()
      dataset_size = len(dataset)
      num_devices = max(1, self.trainer.num_devices)
      num_steps = dataset_size * self.trainer.max_epochs // (self.trainer.accumulate_grad_batches * num_devices)
      return num_steps
  

# training
trainer = pl.Trainer(

                     accumulate_grad_batches=5,                    
                     log_every_n_steps=1,
                     limit_val_batches=2,
                     check_val_every_n_epoch=10,
                     enable_model_summary=True,
                     max_epochs=40, 
                     accelerator='auto',
                     devices='auto',
                     strategy='auto',
                     logger=[TensorBoardLogger("logs/", name="transformer-scratch")],
                     callbacks=[LearningRateMonitor(logging_interval="step")],
                     )
  

def main(config):
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    print(len(train_dataloader.dataset))
    print(len(train_dataloader))
    model = TransformerLightning(config, tokenizer_src, tokenizer_tgt, len(train_dataloader))
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    cfg = get_config()
    cfg['batch_size'] = 40
    cfg['preload'] = None
    cfg['num_epochs'] = 10
    main(cfg)
