from torch import nn
# from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2PreTrainedModel, DebertaV2Model
from transformers.models.bert.modeling_bert import BertPooler
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, DebertaV2Model, DebertaV2Tokenizer, AutoConfig

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CIB_Model(nn.Module):
    def __init__(self, X0_dim, X1_dim, X2_dim, inter_dim, B_dim, output_dim, p_lambda, beta, gamma, sigma, drop_prob):
        super(CIB_Model, self).__init__()
        self.p_lambda = p_lambda
        self.beta = beta
        self.gamma = gamma
        self.sigma = sigma
        self.drop_prob = drop_prob

        self.encoder_v = nn.Sequential(
            nn.Linear(X0_dim, inter_dim),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(inter_dim, B_dim * 2),
        )

        self.encoder_a = nn.Sequential(
            nn.Linear(X1_dim, inter_dim),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(inter_dim, B_dim * 2),
        )

        self.encoder_t = nn.Sequential(
            nn.Linear(X2_dim, inter_dim),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(inter_dim, B_dim * 2),
        )

        # self.fc_fusion = nn.Linear(B_dim * 3, X0_dim)
        # self.fc_out = nn.Linear(X0_dim, output_dim)
        self.criterion = nn.MSELoss()

        self.cond_prob_model_v = ConditionalGaussianEstimator(B_dim, X1_dim, inter_dim, output_dim)
        self.cond_prob_model_a = ConditionalGaussianEstimator(B_dim, X2_dim, inter_dim, output_dim)
        self.cond_prob_model_t = ConditionalGaussianEstimator(B_dim, X0_dim, inter_dim, output_dim)

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
        y = y.expand_as(y_pred).to(y_pred.device)
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

# class VariationalConditionalProbabilityModel(nn.Module):
#     def __init__(self, input_dim_b, input_dim_x, hidden_dim, output_dim):
#         super(VariationalConditionalProbabilityModel, self).__init__()
#         # 编码器部分
#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim_b + input_dim_x, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(0.4),
#             nn.Linear(hidden_dim, output_dim)
#         )
#
#     def forward(self, b, x):
#         combined = torch.cat([b, x], dim=-1)
#         h = self.encoder(combined)
#         return h

class ConditionalGaussianEstimator(nn.Module):
    def __init__(self, input_dim_b0, input_dim_x, hidden_dim, output_dim):
        super(ConditionalGaussianEstimator, self).__init__()
        self.fc1 = nn.Linear(input_dim_b0 + input_dim_x, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, output_dim)
        self.fc_logvar = nn.Linear(hidden_dim, output_dim)

    def forward(self, b0, x1):
        combined = torch.cat([b0, x1], dim=-1)
        h = F.relu(self.fc1(combined))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

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
            nn.Linear(multimodal_config.B_dim * 3, TEXT_DIM),
        )
        self.LayerNorm = nn.LayerNorm(TEXT_DIM)
        self.dropout = nn.Dropout(multimodal_config.dropout_prob)
        self.classifier = nn.Linear(TEXT_DIM, config.num_labels)
        self.init_weights()
        self.pooler = BertPooler(config)

        model_path = "/home/eva/Desktop/deberta-v3-base"
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(model_path)
        self.model = DebertaV2Model.from_pretrained(model_path).to(DEVICE)

        self.mymodel = CIB_Model(VISUAL_DIM, ACOUSTIC_DIM, TEXT_DIM, multimodal_config.inter_dim,
                                 multimodal_config.B_dim, config.num_labels, multimodal_config.p_lambda, multimodal_config.p_beta, multimodal_config.p_gamma, multimodal_config.p_sigma,
                                 multimodal_config.dropout_prob)

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