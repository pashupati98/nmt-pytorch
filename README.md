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
 To be updated further
 
 ![img](save/images/comp-g1-l1.png)
 
 #### Individual performance
 1 : Model with single layer of GRU
 
 Model Summary
 
 ![Model summary (for single layer GRU)](save/images/gru1.PNG)
 
 Training process
 ![training](save/images/g1.png)
 
 Evaluation (Attention Map)
 
 ![attn](save/images/attention_7.png)
 
 2 : Model with single layer of LSTM
 
 Model Summary
 ![summ](save/images/lstm1.PNG)
 
 Training process
 ![train](save/images/l1.png)
 
 Evaluation (Attention Map)
 
 ![attn](save/images/attention_58.png)
 
