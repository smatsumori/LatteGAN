import torch
import torch.nn as nn


class SourceTargetAttention(nn.Module):
    def __init__(
        self,
        query_dim,
        memory_dim,
        hidden_dim,
        use_pos_emb=False,
    ):
        super().__init__()

        self.need_emb_query = query_dim != hidden_dim
        if self.need_emb_query:
            self.emb_query = nn.Sequential(
                nn.Conv2d(query_dim, hidden_dim, 1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(),
            )

        self.attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads=1,
            kdim=memory_dim,
            vdim=memory_dim,
        )
        self.norm = nn.LayerNorm(hidden_dim)

        # positional embedding, buffer 16^2
        self.use_pos_emb = use_pos_emb
        self.pos_emb = nn.Embedding(256, hidden_dim)

    def _add_positional_embedding(self, query):
        query_len, batch_size, _ = query.size()

        pos_ids = torch.arange(0, query_len, device=query.device)
        pos_ids = pos_ids.unsqueeze(dim=0)  # (1, S)
        embed = self.pos_emb(pos_ids)  # (1, S, D)
        embed = embed.repeat(batch_size, 1, 1)  # (B, S, D)
        embed = embed.permute(1, 0, 2).contiguous()  # (S, B, D)

        query = query + embed

        return query

    def _get_key_padding_mask_from_lengths(self, memory, lengths):
        max_seq_len, batch_size, _ = memory.size()

        key_padding_mask = torch.zeros(
            (batch_size, max_seq_len),
            dtype=torch.bool,
            device=memory.device,
        )
        for i, idx in enumerate(lengths):
            key_padding_mask[i, idx:] = True

        return key_padding_mask

    def forward(self, query, memory=None, lengths=None, key_padding_mask=None):
        if self.need_emb_query:
            query = self.emb_query(query)

        b, c, h, w = query.size()
        query = query.view(b, c, h * w)
        query = query.permute(2, 0, 1).contiguous()  # (HW, B, C)
        memory = memory.permute(1, 0, 2).contiguous()  # (L, B, D)

        if (lengths is not None) and (key_padding_mask is None):
            key_padding_mask = \
                self._get_key_padding_mask_from_lengths(memory, lengths)

        if self.use_pos_emb:
            query = self._add_positional_embedding(query)

        output = self.attn(query, memory, memory,
                           key_padding_mask=key_padding_mask)[0]
        output = self.norm(output)

        output = output.permute(1, 2, 0).contiguous()
        output = output.view(b, c, h, w)  # (B, C, H, W)

        return output


class TIRG(nn.Module):
    def __init__(
        self,
        image_dim,
        text_dim,
        hidden_dim,
        sa_fused=False,
        sta_concat=False,
        use_pos_emb=False,
        sa_gate=False,
        res_mask=False,
        res_mask_post=False,
        use_conv_final=False,
        multi_channel_gate=False,
        num_objects=58,
    ):
        super().__init__()

        self.w_gate = nn.Parameter(torch.ones(1))
        self.w_res = nn.Parameter(torch.zeros(1))

        # concat --> fusion
        self.sa_fused = sa_fused
        if sa_fused:
            self.conv_fused = nn.Sequential(
                nn.Conv2d(image_dim + text_dim, hidden_dim, 3, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(),
            )
            self.attn_fused = nn.MultiheadAttention(hidden_dim, num_heads=1)
            self.bn_attn_fused = nn.BatchNorm2d(hidden_dim)
            self.relu_attn_fused = nn.ReLU()

        if sa_fused:
            in_channels = hidden_dim
        else:
            in_channels = image_dim + text_dim

        # concat --> source-target-attention
        self.sta_concat = sta_concat
        if sta_concat:
            self.st_attn = SourceTargetAttention(
                in_channels,
                text_dim,
                hidden_dim,
                use_pos_emb,
            )
            in_channels += hidden_dim

        # gating 1/2
        self.conv_gate1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
        )
        self.sa_gate = sa_gate
        if sa_gate:
            self.attn_gate = nn.MultiheadAttention(hidden_dim, num_heads=1)
            self.bn_attn_gate = nn.BatchNorm2d(hidden_dim)
            self.relu_attn_gate = nn.ReLU()

        # gating 2/2 (single or multi channels)
        self.multi_channel_gate = multi_channel_gate
        if multi_channel_gate:
            gate_dim = image_dim
            gate_detect_dim = image_dim
        else:
            gate_dim = 1
            gate_detect_dim = hidden_dim
        self.conv_gate2 = nn.Sequential(
            nn.Conv2d(hidden_dim, gate_dim, 3, padding=1),
            nn.Sigmoid(),
        )

        # detection added objects from gating pass
        # single channels gate --> branch gating 1/2
        # multi channels gate --> branch gating 2/2
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.detector = nn.Linear(gate_detect_dim, num_objects)

        # residual
        self.conv_res = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, image_dim, 3, padding=1),
            nn.BatchNorm2d(image_dim),
            nn.ReLU(),
        )
        self.res_mask = res_mask
        self.res_mask_post = res_mask_post

        # final modification
        self.use_conv_final = use_conv_final
        if use_conv_final:
            self.conv_final = nn.Sequential(
                nn.Conv2d(image_dim, image_dim, 1),
                nn.BatchNorm2d(image_dim),
                nn.ReLU(),
            )

    def _forward_attn_fused(self, x):
        b, c, h, w = x.size()

        x = x.view(b, c, h * w).permute(2, 0, 1).contiguous()  # (HW, B, C)
        x = self.attn_fused(x, x, x)[0]  # (HW, B, C)
        x = x.permute(1, 2, 0).contiguous().view(b, c, h, w)  # (B, C, H, W)
        x = self.bn_attn_fused(x)  # (B, C, H, W)
        x = self.relu_attn_fused(x)  # (B, C, H, W)

        return x

    def _forward_attn_gate(self, x):
        b, c, h, w = x.size()

        x = x.view(b, c, h * w).permute(2, 0, 1).contiguous()  # (HW, B, C)
        x = self.attn_gate(x, x, x)[0]  # (HW, B, C)
        x = x.permute(1, 2, 0).contiguous().view(b, c, h, w)  # (B, C, H, W)
        x = self.bn_attn_gate(x)  # (B, C, H, W)
        x = self.relu_attn_gate(x)  # (B, C, H, W)

        return x

    def forward(
        self, x_img, x_txt, memory=None, lengths=None, key_padding_mask=None
    ):
        b, _, h, w = x_img.size()
        x_txt = x_txt.view(b, -1, 1, 1).repeat(1, 1, h, w)
        x = torch.cat([x_img, x_txt], dim=1)

        # fused
        if self.sa_fused:
            x = self.conv_fused(x)
            x = self._forward_attn_fused(x)

        # source target concat
        if self.sta_concat:
            y_txt = self.st_attn(x, memory, lengths, key_padding_mask)
            x = torch.cat([x, y_txt], dim=1)

        # gating 1/2
        gate1 = self.conv_gate1(x)
        if self.sa_gate:
            gate1 = self._forward_attn_gate(gate1)

        # gating 2/2
        gate2 = self.conv_gate2(gate1)

        # added object detection
        if self.multi_channel_gate:
            detect_from_gate = self.pool(gate2).squeeze(dim=3).squeeze(dim=2)
        else:
            detect_from_gate = self.pool(gate1).squeeze(dim=3).squeeze(dim=2)
        detect_from_gate = self.detector(detect_from_gate)

        # residual
        if self.res_mask:
            # error when multi channels gate
            x = (1 - gate2) * x
        res = self.conv_res(x)
        if self.res_mask_post:
            res = (1 - gate2) * res

        y = self.w_gate * (gate2 * x_img) + self.w_res * res
        if self.use_conv_final:
            y = self.conv_final(y)

        return y, gate2, detect_from_gate
