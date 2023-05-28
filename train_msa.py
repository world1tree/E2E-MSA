import torch
from torch.optim import Adam
from build_dataset import build_data_iter
from build_model import build_model
from dataclasses import dataclass, field

@dataclass
class ModelConfig(object):
    src_word_vec_size:int=field(
        default=512,
        metadata={
            "help": "Word embedding size for src."
        }
    )
    tgt_word_vec_size:int=field(
        default=512,
        metadata={
            "help": "Word embedding size for tgt."
        }
    )
    share_embeddings:bool=field(
        default=True,
        metadata={
            "help": "Share the word embeddings between encoder "
                    "and decoder. Need to use shared dictionary for this "
                    "option."
        }
    )
    position_encoding:bool=field(
        default=True,
        metadata={
            "help": "Use a sin to mark relative words positions."
                    "Necessary for non-RNN style models."
        }
    )
    enc_layers:int=field(
        default=1,
        metadata={
            "help": "Number of layers in the encoder"
        }
    )
    dec_layers:int=field(
        default=1,
        metadata={
            "help": "Number of layers in the decoder"
        }
    )
    enc_rnn_size:int=field(
        default=512,
        metadata={
            "help": "Size of encoder rnn hidden states."
                    "Must be equal to dec_rnn_size except for"
                    "speech-to-text."
        }
    )
    dec_rnn_size:int=field(
        default=512,
        metadata={
            "help": "Size of decoder rnn hidden states."
                    "Must be equal to dec_rnn_size except for"
                    "speech-to-text."
        }
    )
    self_attn_type:str=field(
        default="scaled-dot",
        metadata={
            "help": "Self attention type in Transformer decoder"
                    "layer -- currently 'scaled-dot' or 'average'"
        }
    )

    heads:int=field(
        default=8,
        metadata={
            "help": "Number of heads for transformer self-attention"
        }
    )

    transformer_ff:int=field(
        default=2048,
        metadata={
            "help": "Size of hidden transformer feed-forward"
        }
    )

    dropout:float=field(
        default=0.1,
        metadata={
            "help": "Dropout probability; applied in LSTM stacks."
        }
    )

def train(model, train_dataloader, val_dataloader, learning_rate, epochs):
    padding_idx = model.encoder.padding_idx
    # 通过Dataset类获取训练和验证集
    # 判断是否使用GPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # 定义损失函数和优化器
    criterion = torch.nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()
    # 开始进入训练循环
    for epoch_num in range(epochs):
        # 定义两个变量，用于存储训练集的准确率和损失
        total_acc1_train = 0
        total_acc2_train = 0
        total_loss_train = 0
        # 进度条函数tqdm
        for train_data in train_dataloader:
            seq1 = train_data["seq1"]
            seq2 = train_data["seq2"]
            length1 = train_data["length1"]
            length2 = train_data["length2"]
            label1 = train_data["label1"]
            label2 = train_data["label2"]
            # 通过模型得到输出
            output1 = model(seq1, seq2, length1)[0]
            label2_logits = model.generator(output1)
            label2_pred = torch.argmax(label2_logits, dim=-1)
            seq2_mask = (~(seq2.eq(padding_idx))).to(torch.float)
            label2_pred = label2_pred * seq2_mask
            loss2 = criterion(label2, label2_pred)
            acc2 = torch.sum((label2==label2_pred)*seq2_mask, dtype=torch.long).item()
            total2 = torch.sum((label2!=label2_pred)*seq2_mask, dtype=torch.long).item()

            output2 = model(seq2, seq1, length2)[0]
            label1_logits = model.generator(output2)
            label1_pred = torch.argmax(label1_logits, dim=-1)
            seq1_mask = (~(seq1.eq(padding_idx))).to(torch.float)
            label1_pred = label1_pred * seq1_mask
            loss1 = criterion(label1, label1_pred)
            acc1 = torch.sum((label1==label1_pred)*seq1_mask, dtype=torch.long).item()
            total1 = torch.sum((label1!=label1_pred)*seq1_mask, dtype=torch.long).item()

            batch_loss = loss1 + loss2
            # 计算损失
            total_loss_train += batch_loss.item()
            # 计算精度
            total_acc2_train += acc2
            total_acc1_train += acc1
            # 模型更新
            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        # # ------ 验证模型 -----------
        # # 定义两个变量，用于存储验证集的准确率和损失
        # total_acc_val = 0
        # total_loss_val = 0
        # # 不需要计算梯度
        # with torch.no_grad():
        #     # 循环获取数据集，并用训练好的模型进行验证
        #     for val_input, val_label in val_dataloader:
        #         # 如果有GPU，则使用GPU，接下来的操作同训练
        #         val_label = val_label.to(device)
        #         mask = val_input['attention_mask'].to(device)
        #         input_id = val_input['input_ids'].squeeze(1).to(device)
        #
        #         output = model(input_id, mask)
        #
        #         batch_loss = criterion(output, val_label)
        #         total_loss_val += batch_loss.item()
        #
        #         acc = (output.argmax(dim=1) == val_label).sum().item()
        #         total_acc_val += acc

        print(
            f'''Epochs: {epoch_num + 1} 
              | Train Loss: {total_loss_train / len(train_data): .3f} 
              | Train Accuracy1: {total_acc2_train / len(train_data): .3f} 
              | Train Accuracy2: {total_acc1_train / len(train_data): .3f}''')
              # | Val Loss: {total_loss_val / len(val_data): .3f}
              # | Val Accuracy: {total_acc_val / len(val_data): .3f}''')

EPOCHS = 5
model = build_model(ModelConfig())
LR = 1e-6
train_dataloader = build_data_iter(["data/train_0.txt", "data/train_1.txt"], data_type="train")
valid_dataloader = build_data_iter(["data/valid_0.txt", "data/valid_1.txt"], data_type="valid")
train(model, train_dataloader, valid_dataloader, LR, EPOCHS)
