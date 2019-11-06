import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import src.utils.init as my_init
from src.utils import nest


def sort_batch(seq_len):
    """Sorts torch tensor of integer indices by decreasing order."""

    # oidx  slens到seq_len的下标对应，slens[oidx[i]] = slens[i]
    # sidx seq_len到slens的下标对应，seq_len[sidx[i]] = slens[i]

    with torch.no_grad():
        # slens: 排好序的
        # sidx: 排好序的 在原数组中的位置
        slens, sidxs = torch.sort(seq_len, descending=True)

    # 对索引进行排序，学到了
    oidxs = torch.sort(sidxs)[1]

    return oidxs, sidxs, slens.tolist()


class RNN(nn.Module):

    def __init__(self, type, batch_first=False, **kwargs):

        super().__init__()

        self.type = type
        self.batch_first = batch_first

        # We always use batch first mode.
        if self.type == "gru":
            self.rnn = nn.GRU(batch_first=True, **kwargs)
        elif self.type == "lstm":
            self.rnn = nn.LSTM(batch_first=batch_first, **kwargs)

        self._reset_parameters()

    @property
    def batch_dim(self):
        if self.batch_first:
            return 0
        else:
            return 1

    def _reset_parameters(self):
        for weight in self.rnn.parameters():
            my_init.rnn_init(weight)

    def forward(self, input, input_mask, h_0=None):
        """
        :param input: Input sequence.
            With shape [batch_size, input_len, dim] if batch_first is True.

        :param input_mask: Mask of sequence. [batch_size, input_len]
        """

        self.rnn.flatten_parameters()  # This is necessary if want to use DataParallel

        # Convert into batch first
        if self.batch_first is False:
            input = input.transpose(0, 1).contiguous()
            input_mask = input_mask.transpose(0, 1).contiguous()

        ##########################
        # Pad zero length with 1 #
        ##########################
        with torch.no_grad():
            seq_len = (1 - input_mask.long()).sum(1)  # [batch_size]
            # 长度为0的句子设置为长度为1？？？  这个代码会被执行到吗
            seq_len[seq_len.eq(0)] = 1
        # out: [batch_size, seq_len, hid_dim*num_directions]
        # h_n: [n_layers * num_directions, batch_size. hid_dim]
        out, h_n = self._forward_rnn(input, seq_len, h_0=h_0)

        if self.batch_first is False:
            out = out.transpose(0, 1).contiguous()  # Convert to batch_second

        return out, h_n

    def _forward_rnn(self, input, input_length, h_0=None):
        """
        :param input: Input sequence.
            FloatTensor with shape [batch_size, input_len, dim]

        :param input_length: Mask of sequence.
            LongTensor with shape [batch_size, ]
        """
        # 一个batch里的一条为一个句子，batch内部排序是按照长度更改他在batch中的下标，所以是在batch维度进行变化的
        #
        total_length = input.size(1)

        # 1. Packed with pad
        # oidx  slens到seq_len的下标对应，slens[oidx[i]] = slens[i]
        # sidx seq_len到slens的下标对应，seq_len[sidx[i]] = slens[i]
        # slens 排好序的结果
        oidx, sidx, slens = sort_batch(input_length)

        # 其实按照sidx的次序重新排序input_sorted, 这样input_sorted和slens的下标是对应的
        input_sorted = torch.index_select(input, index=sidx, dim=0)

        # h_0是隐层输入, 这是说h_0 也需要排序？？
        if h_0 is not None:
            h_0_sorted = nest.map_structure(lambda t: torch.index_select(t, 1, sidx), h_0)
        else:
            h_0_sorted = None

        # 2. RNN compute
        # 为了节省资源，按照读取时间的维度打平成二维数据，每一个时间步batch_size可以不一样
        input_packed = pack_padded_sequence(input_sorted, slens, batch_first=True)

        # out_packed: [batch_size, seq_len, hid_dim * num_directions]
        # h_n_sorted: [n_layers * num_directions, batch_size, hid_dim]
        out_packed, h_n_sorted = self.rnn(input_packed, h_0_sorted)

        # 3. Restore
        # https://zhuanlan.zhihu.com/p/59772104
        # 在把0填充上，整成三维的格式
        out_sorted = pad_packed_sequence(out_packed, batch_first=True, total_length=total_length)[0]
        # 在排序回去 out:[batch_size, seq_len, hid_dim*num_directions]
        out = torch.index_select(out_sorted, dim=0, index=oidx)
        # 相当于torch.index(h_n_sorted, 1, oidx)
        h_n_sorted = nest.map_structure(lambda t: torch.index_select(t, 1, oidx), h_n_sorted)

        return out.contiguous(), h_n_sorted
