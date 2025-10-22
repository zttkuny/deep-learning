from Encoder import TransformerEncoder
from Decoder import TransformerDecoder
from Decoder_Encoder.main import *



num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 10
lr, num_epochs, device = 0.005, 200, d2l.try_gpu()
ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
key_size, query_size, value_size = 32, 32, 32
norm_shape = [32]
train_iter, src_vocab, tgt_vocab = trans.load_data_nmt(batch_size, num_steps,100000)

encoder=TransformerEncoder(vocab_size=len(src_vocab),query_size=num_hiddens,key_size=num_hiddens,value_size=num_hiddens,num_hiddens=num_hiddens,
                           num_heads=num_heads,num_layers=num_layers,num_FFN_inputs=ffn_num_input,num_FFN_hiddens=ffn_num_hiddens,num_FFN_output=ffn_num_input,
                           dropout=dropout,norm_size=norm_shape)

decoder=TransformerDecoder(vocab_size=len(tgt_vocab),query_size=num_hiddens,key_size=num_hiddens,value_size=num_hiddens,num_hiddens=num_hiddens,
                           num_heads=num_heads,num_layers=num_layers,ffn_num_inputs=ffn_num_input,ffn_num_hiddens=ffn_num_hiddens,
                           dropout=dropout,norm_shape=norm_shape)



net = d2l.EncoderDecoder(encoder, decoder)

train_seq2seq(net,train_iter,lr,num_epochs,tgt_vocab,device)