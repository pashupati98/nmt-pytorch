# nmt-pytorch
Pytorch based implementation of a seq2seq machine translation model. This implementaton is based on two following papers.
- Ilya Sutskever, Oriol Vinyals, Quoc V. Le **Sequence to sequence learning with neural networks**
- Dzmitry Bahdanau, KyungHyun Cho, Yoshua Bengio∗ (2016) **Neural machine translation by jointly learning to align and translate** 

#### Dataset Description

The data for this project is a set of many thousands translation pairs from one language to another. Download the data from here (https://www.manythings.org/anki/)

#### Hardware Configuration

Training was performed on Google Colaboratory platform which provides free access to GPUs. GPU Config -
Tesla P100-PCIE-16GB having 2496 CUDA cores and 16GB GDDR5 VRAM.

#### Model Description

Encoder-Decoder with attention mechanism
 
 Model variants (architecture of both encoder and decoder)
 - Single layer GRU (23M paremeters)
 - Single layer LSTM
 - 5-layered GRU
 - 5-layered LSTM
 
 #### Comperative performance 
 To be updated
 
 #### Individual performance
 Model with single layer GRU
 
 [Model summary (for single layer GRU)](save/images/gru1.PNG)
 
 Training
 Evaluation
