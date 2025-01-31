from torch import nn
# from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2PreTrainedModel, DebertaV2Model
from transformers.models.bert.modeling_bert import BertPooler
import torch
import math
import torch.nn.functional as F
from transformers import PreTrainedModel, DebertaV2Model, DebertaV2Tokenizer, AutoConfig

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 位置编码类
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


# Transformer-based Conditional Estimator
class ConditionalTransformerEstimator(nn.Module):
    def __init__(self, input_dim_x, B0_dim, B1_dim, nhead, num_layers, output_dim):
        super(ConditionalTransformerEstimator, self).__init__()
        self.input_dim = int(B0_dim / 2 + input_dim_x)
        self.fc_in = nn.Linear(self.input_dim, B1_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=B1_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc_mu = nn.Linear(B1_dim, output_dim)
        self.fc_logvar = nn.Linear(B1_dim, output_dim)

    def forward(self, b0, x1):
        combined = torch.cat([b0, x1], dim=-1) # 不用！添加一个额外的维度用于 Transformer
        combined = self.fc_in(combined)

        # Transformer Encoder expects input shape as [seq_len, batch_size, d_model]
        combined = combined.permute(1, 0, 2)
        transformed = self.transformer_encoder(combined)

        # We only need the output corresponding to the first sequence (sequence-to-sequence model)
        transformed = transformed.mean(dim=0)

        mu = self.fc_mu(transformed)
        logvar = self.fc_logvar(transformed)
        return mu, logvar


# Transformer-based encoder for each modality
class TransformerEncoderModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, nhead, num_layers, drop_prob):
        super(TransformerEncoderModule, self).__init__()
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # We no longer need to reduce the sequence length, so no final linear layer
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        # x shape is [batch_size, seq_length, input_dim]
        x = self.fc_in(x)  # Linear layer: [batch_size, seq_length, input_dim] -> [batch_size, seq_length, hidden_dim]
        x = x.permute(1, 0, 2)  # Permute to [seq_length, batch_size, hidden_dim]
        x = self.pos_encoder(x)  # Add positional encoding: [seq_length, batch_size, hidden_dim]
        x = self.transformer_encoder(x)  # Apply Transformer: [seq_length, batch_size, hidden_dim]
        x = x.permute(1, 0, 2)  # Permute back to [batch_size, seq_length, hidden_dim]
        x = self.dropout(x)  # Apply dropout
        return x  # Output shape: [batch_size, seq_length, hidden_dim]

class CIB_Model(nn.Module):
    def __init__(self, X0_dim, X1_dim, X2_dim, B0_dim, B1_dim, output_dim, p_lambda, beta, gamma, sigma, drop_prob, nhead, num_layers):
        super(CIB_Model, self).__init__()
        self.p_lambda = p_lambda
        self.beta = beta
        self.gamma = gamma
        self.sigma = sigma
        self.drop_prob = drop_prob

        self.encoder_v = TransformerEncoderModule(X0_dim, B0_dim, nhead, num_layers, drop_prob)
        self.encoder_a = TransformerEncoderModule(X1_dim, B0_dim, nhead, num_layers, drop_prob)
        self.encoder_t = TransformerEncoderModule(X2_dim, B0_dim, nhead, num_layers, drop_prob)

        self.criterion = nn.MSELoss()

        # Replace MLP-based conditional estimators with Transformer-based ones
        self.cond_prob_model_v = ConditionalTransformerEstimator(X1_dim, B0_dim, B1_dim, nhead, num_layers, output_dim)
        self.cond_prob_model_a = ConditionalTransformerEstimator(X2_dim, B0_dim, B1_dim, nhead, num_layers, output_dim)
        self.cond_prob_model_t = ConditionalTransformerEstimator(X0_dim, B0_dim, B1_dim, nhead, num_layers, output_dim)

    def kl_loss(self, mu, logvar):
        kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        kl_mean = torch.mean(kl_div)
        return kl_mean

    def reparameterise(self, mu, logvar):
        epsilon = torch.randn_like(mu)
        return mu + epsilon * torch.exp(logvar / 2)

    def variational_conditional_loss(self, conditional_estimator, b, y, x):
        mu, logvar = conditional_estimator(b, x)
        # y_pred = y_pred.mean(dim=1, keepdim=True)  # 将其平均到 [batch_size, 1, 1]
        # recon_loss = self.criterion(y_pred, y)
        # return recon_loss
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        y_pred = mu + epsilon * std

        # 损失计算
        # y = y.expand_as(y_pred).to(y_pred.device)
        y = y.squeeze(dim=-1)
        reconstruction_loss = self.criterion(y_pred, y)  # 重构损失
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)  # KL 散失

        total_loss = reconstruction_loss + kl_loss.mean()
        return total_loss

    def forward(self, visual, acoustic, text, labels):
        # Video modality
        h_v = self.encoder_v(visual)
        mu_v, logvar_v = h_v.chunk(2, dim=-1)
        kl_loss_v = self.kl_loss(mu_v, logvar_v)
        b_v = self.reparameterise(mu_v, logvar_v)

        # Audio modality
        h_a = self.encoder_a(acoustic)
        mu_a, logvar_a = h_a.chunk(2, dim=-1)
        kl_loss_a = self.kl_loss(mu_a, logvar_a)
        b_a = self.reparameterise(mu_a, logvar_a)

        # Text modality
        h_t = self.encoder_t(text)
        mu_t, logvar_t = h_t.chunk(2, dim=-1)
        kl_loss_t = self.kl_loss(mu_t, logvar_t)
        b_t = self.reparameterise(mu_t, logvar_t)

        # Conditional losses
        cond_loss_v = self.variational_conditional_loss(self.cond_prob_model_v, b_v, labels, acoustic)
        cond_loss_a = self.variational_conditional_loss(self.cond_prob_model_a, b_a, labels, text)
        cond_loss_t = self.variational_conditional_loss(self.cond_prob_model_t, b_t, labels, visual)

        # Total loss
        total_loss_v = kl_loss_v + cond_loss_v
        total_loss_a = kl_loss_a + cond_loss_a
        total_loss_t = kl_loss_t + cond_loss_t
        total_loss = self.beta * total_loss_v + self.gamma * total_loss_a + self.sigma * total_loss_t

        # Fusion
        output = torch.cat([b_v, b_a, b_t], dim=-1)

        return output, total_loss, kl_loss_v, cond_loss_v, kl_loss_a, cond_loss_a, kl_loss_t, cond_loss_t


class CIBForSequenceClassification(PreTrainedModel):
    config_class = AutoConfig
    base_model_prefix = "deberta"

    def __init__(self, config, multimodal_config):
        super().__init__(config)
        TEXT_DIM, ACOUSTIC_DIM, VISUAL_DIM = (768, 291, 2048)
        self.config = config
        self.multimodal_config = multimodal_config

        self.num_labels = config.num_labels
        self.expand = nn.Sequential(
            nn.Linear(int(multimodal_config.B0_dim * 3 / 2), TEXT_DIM),
        )
        self.LayerNorm = nn.LayerNorm(TEXT_DIM)
        self.dropout = nn.Dropout(multimodal_config.dropout_prob)
        self.classifier = nn.Linear(TEXT_DIM, config.num_labels)
        self.init_weights()
        self.pooler = BertPooler(config)

        model_path = "/home/eva/Desktop/deberta-v3-base"
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(model_path)
        self.model = DebertaV2Model.from_pretrained(model_path).to(DEVICE)

        self.mymodel = CIB_Model(VISUAL_DIM, ACOUSTIC_DIM, TEXT_DIM, multimodal_config.B0_dim,
                                 multimodal_config.B1_dim, config.num_labels, multimodal_config.p_lambda,
                                 multimodal_config.p_beta, multimodal_config.p_gamma, multimodal_config.p_sigma,
                                 multimodal_config.dropout_prob, multimodal_config.multi_head,
                                 multimodal_config.num_layers)

    def forward(self, visual, acoustic, input_ids, labels):
        input_ids = input_ids.to(DEVICE)
        visual = visual.to(DEVICE)
        acoustic = acoustic.to(DEVICE)
        labels = labels.to(DEVICE)

        embedding_output = self.model(input_ids)
        x = embedding_output[0]

        batch_size, seq_length, hidden_dim = x.size()
        new_seq_length = self.multimodal_config.max_seq_length
        pooled_x = torch.zeros(batch_size, new_seq_length, hidden_dim).to(DEVICE)
        for i in range(new_seq_length):
            start_idx = i * (seq_length // new_seq_length)
            end_idx = (i + 1) * (seq_length // new_seq_length)
            if i == new_seq_length - 1:
                end_idx = seq_length
            pooled_x[:, i, :] = torch.mean(x[:, start_idx:end_idx, :], dim=1)

        output, total_loss, kl_loss_v, cond_loss_v, kl_loss_a, cond_loss_a, kl_loss_t, cond_loss_t = self.mymodel(
            visual, acoustic, pooled_x, labels
        )

        all_output = self.dropout(self.LayerNorm(self.expand(output) + x))
        pooled_output = self.pooler(all_output)
        logits = pooled_output.view(pooled_output.size(0), -1)
        # logits = self.dropout(logits)
        logits = self.classifier(logits)
        # logits = torch.sigmoid(logits)  # 使用sigmoid将输出转换为概率值

        return logits, total_loss, kl_loss_v, cond_loss_v, kl_loss_a, cond_loss_a, kl_loss_t, cond_loss_t