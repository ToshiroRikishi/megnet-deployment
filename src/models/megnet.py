import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# =============================================================================
# Вспомогательные классы для MegaNet
# =============================================================================
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_output, _ = self.multihead_attn(x, x, x)
        return self.layer_norm(x + attn_output)

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super(ResidualBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout)
        )
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.layer_norm(x + self.layers(x))

# =============================================================================
# Основной класс модели MegaNet
# =============================================================================
class MegaNet(nn.Module):
    def __init__(self, input_size, num_classes, params):
        super(MegaNet, self).__init__()
        shared_embed_dim = params['shared_embed_dim']
        latent_dim = params['latent_dim']
        num_heads = params['num_heads']
        dropout = params['dropout']
        self.num_classes = num_classes

        # 1) Общий входной эмбеддинг
        self.shared_input_embed = nn.Sequential(
            nn.Linear(input_size, shared_embed_dim),
            nn.LayerNorm(shared_embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 2) Трансформерная ветка
        self.trans_path = nn.Sequential(
            MultiHeadSelfAttention(shared_embed_dim, num_heads),
            ResidualBlock(shared_embed_dim, dropout)
        )

        # 3) VAE ветка
        self.vae_encoder_mu = nn.Linear(shared_embed_dim, latent_dim)
        self.vae_encoder_logvar = nn.Linear(shared_embed_dim, latent_dim)
        self.vae_decoder = nn.Linear(latent_dim, input_size)
        self.vae_transformer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=max(1, num_heads // 2),
            batch_first=True
        )

        # 4) Квантовая ветка
        num_states = 4
        self.quan_superposition = nn.Linear(shared_embed_dim, shared_embed_dim * num_states)
        self.quan_entanglement = nn.TransformerEncoderLayer(
            d_model=shared_embed_dim,
            nhead=max(1, num_heads // 2),
            batch_first=True
        )
        self.quan_measurement = nn.Linear(shared_embed_dim * num_states, shared_embed_dim)

        # 5) Эволюционные ветки
        self.evo_path1 = MultiHeadSelfAttention(shared_embed_dim, max(1, num_heads // 2))
        self.evo_path2 = ResidualBlock(shared_embed_dim)
        self.evo_gates = nn.Parameter(torch.ones(2))

        # 6) Диффузионная ветка
        self.diffu_path = nn.Sequential(
            ResidualBlock(shared_embed_dim, dropout),
            ResidualBlock(shared_embed_dim, dropout)
        )

        # 7) Слой объединения признаков и классификатор
        fusion_input_dim = shared_embed_dim + latent_dim + shared_embed_dim + shared_embed_dim + shared_embed_dim
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_input_dim // 2),
            nn.BatchNorm1d(fusion_input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # Для бинарной классификации используем 1 выходной нейрон
        output_dim = 1 if num_classes == 2 else num_classes
        self.classifier = nn.Linear(fusion_input_dim // 2, output_dim)

        # Уровни шума (только для train)
        self.noise_levels = [0.05, 0.15]

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_original):
        if self.training:
            noise = torch.randn_like(x_original) * np.random.choice(self.noise_levels)
            x = self.shared_input_embed(x_original + noise)
        else:
            x = self.shared_input_embed(x_original)

        # Transformer path
        out1 = self.trans_path(x.unsqueeze(1)).squeeze(1)

        # VAE path
        mu = self.vae_encoder_mu(x)
        logvar = self.vae_encoder_logvar(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.vae_decoder(z)
        out2 = self.vae_transformer(z.unsqueeze(1)).squeeze(1)

        # Quantum path
        b, _ = x.shape
        quan_states = self.quan_superposition(x).view(b, 4, -1)
        quan_entangled = self.quan_entanglement(quan_states)
        out3 = self.quan_measurement(quan_entangled.view(b, -1))

        # Evolution path
        g = F.softmax(self.evo_gates, dim=0)
        evo_out1 = self.evo_path1(x.unsqueeze(1)).squeeze(1)
        evo_out2 = self.evo_path2(x)
        out4 = g[0] * evo_out1 + g[1] * evo_out2

        # Diffusion path
        out5 = self.diffu_path(x)

        # Fusion
        concatenated = torch.cat([out1, out2, out3, out4, out5], dim=1)
        fused = self.fusion_layer(concatenated)
        logits = self.classifier(fused)
        
        # Возвращаем правильную размерность для бинарной классификации
        if self.num_classes == 2:
            logits = logits.squeeze(-1)
        
        return logits, recon_x, mu, logvar

# =============================================================================
# Обёртка для детерминированного инференса
# =============================================================================
class MegaNetInference(nn.Module):
    def __init__(self, base_model: MegaNet):
        super(MegaNetInference, self).__init__()
        self.base = base_model
        self.base.eval()
        # Отключаем все Dropout
        for m in self.base.modules():
            if isinstance(m, nn.Dropout):
                m.p = 0.0
        # Заменяем sampling на прямое mu
        self.base.reparameterize = lambda mu, logvar: mu

    def forward(self, x):
        x_emb = self.base.shared_input_embed(x)
        out1 = self.base.trans_path(x_emb.unsqueeze(1)).squeeze(1)
        mu = self.base.vae_encoder_mu(x_emb)
        out2 = self.base.vae_transformer(mu.unsqueeze(1)).squeeze(1)

        b, _ = x_emb.shape
        quan_states = self.base.quan_superposition(x_emb).view(b, 4, -1)
        quan_entangled = self.base.quan_entanglement(quan_states)
        out3 = self.base.quan_measurement(quan_entangled.view(b, -1))

        g = F.softmax(self.base.evo_gates, dim=0)
        evo_out1 = self.base.evo_path1(x_emb.unsqueeze(1)).squeeze(1)
        evo_out2 = self.base.evo_path2(x_emb)
        out4 = g[0] * evo_out1 + g[1] * evo_out2

        out5 = self.base.diffu_path(x_emb)
        concatenated = torch.cat([out1, out2, out3, out4, out5], dim=1)
        fused = self.base.fusion_layer(concatenated)
        logits = self.base.classifier(fused)
        return logits