from Encoder import *
import  MachineTrans.Trans as trans
def predict_seq_seq(net,src_sentence,src_vocab,tgt_vocab,num_steps,device):
    net.eval() #开启推理模式

    src_tokens=src_sentence.lower().split(' ')+['<eos>']
    src_tokens=[src_vocab[s] for s in src_tokens]

    src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab['<pad>']) # 固定长度的序列

    src_tokens=torch.tensor(src_tokens,device=device)
    X=src_tokens.unsqueeze(0) # 添加批次维度

    out,state=net.Encoder(X) # state是编码器最后一个时间步的隐藏状态输出

    dec_input=torch.tensor([tgt_vocab['<bos>']],device=device).unsqueeze(0) # 添加批次维度

    output_seq=[]

    for i in range(num_steps):
        out,state=net.Decoder(dec_input,state)
        dec_input=out.argmax(dim=2)
        token_id=dec_input.squeeze(0).type(torch.int).item() #最后得到预测概率最高的词索引

        if token_id==tgt_vocab['<eos>']:
            break

        output_seq.append(tgt_vocab.to_tokens(token_id))

    return ' '.join(output_seq)

train_iter, src_vocab, tgt_vocab = trans.load_data_nmt(batch_size=64, num_steps=10, num_examples=100000)
device=torch.device('cuda')

net=EncoderDecoder(len(src_vocab),32,len(tgt_vocab),32,512,2,drop_out=0.5)
net.to(device)

checkpoint=torch.load('check_point_75.pth')
net.load_state_dict(checkpoint['state_dict'])

text=predict_seq_seq(net,'waht\'s your name ?',src_vocab,tgt_vocab,10,device)
print(text)



