import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from scipy import stats
import matplotlib
from matplotlib.lines import Line2D
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
from scipy.stats import gaussian_kde
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ConditionalVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=64, hidden_dim=128, num_classes=2):
        super(ConditionalVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_classes, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh()
        )
    def encode(self, x, labels):
        labels_onehot = F.one_hot(labels.long(), self.num_classes).float()
        x_labeled = torch.cat([x, labels_onehot], dim=1)
        h = self.encoder(x_labeled)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    def decode(self, z, labels):
        labels_onehot = F.one_hot(labels.long(), self.num_classes).float()
        z_labeled = torch.cat([z, labels_onehot], dim=1)
        return self.decoder(z_labeled)
    def forward(self, x, labels):
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, labels)
        return recon_x, mu, logvar
    def sample(self, num_samples, labels, device='cpu'):
        self.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(device)
            labels = labels.to(device)
            generated = self.decode(z, labels)
        return generated
class ConditionalVAE1(nn.Module):

    def __init__(self, input_dim, latent_dim=64, hidden_dims=[512, 256, 128], num_classes=2):
        super(ConditionalVAE1, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.input_norm = nn.LayerNorm(input_dim + num_classes)
        encoder_layers = []
        in_dim = input_dim + num_classes

        for i, hidden_dim in enumerate(hidden_dims):
            encoder_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ELU(),
                nn.Dropout(0.1)
            ])
            in_dim = hidden_dim

        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        decoder_layers = []
        in_dim = latent_dim + num_classes

        for i, hidden_dim in enumerate(reversed(hidden_dims)):
            decoder_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ELU(),
                nn.Dropout(0.1)
            ])
            in_dim = hidden_dim
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            torch.nn.init.constant_(m.bias, 0)

    def encode(self, x, labels):
        labels_onehot = F.one_hot(labels.long(), self.num_classes).float()
        x_labeled = torch.cat([x, labels_onehot], dim=1)
        x_labeled = self.input_norm(x_labeled)
        h = self.encoder(x_labeled)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, labels):
        labels_onehot = F.one_hot(labels.long(), self.num_classes).float()
        z_labeled = torch.cat([z, labels_onehot], dim=1)
        return self.decoder(z_labeled)

    def forward(self, x, labels):
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, labels)
        return recon_x, mu, logvar

    def sample(self, num_samples, labels, device='cpu'):
        self.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(device)
            labels = labels.to(device)
            generated = self.decode(z, labels)
        return generated

class VAEEvaluator:
    def __init__(self):
        self.training_losses = []

    def evaluate_reconstruction_quality(self, vae, original_data, labels, device):
        vae.eval()
        with torch.no_grad():
            X = torch.FloatTensor(original_data).to(device)
            y = torch.LongTensor(labels).to(device)
            recon_x, mu, logvar = vae(X, y)
            recon_error = F.mse_loss(recon_x, X, reduction='none').mean(dim=1)
            mean_recon_error = recon_error.mean().item()
            X_np = X.cpu().numpy()
            recon_x_np = recon_x.cpu().numpy()
            correlations = []
            for i in range(X_np.shape[1]):
                corr = np.corrcoef(X_np[:, i], recon_x_np[:, i])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
            mean_correlation = np.mean(correlations)
            min_correlation = np.min(correlations)
            return mean_recon_error, mean_correlation

    def compare_distributions(self, original_data, generated_data):
        orig_mean = np.mean(original_data, axis=0)
        gen_mean = np.mean(generated_data, axis=0)
        orig_std = np.std(original_data, axis=0)
        gen_std = np.std(generated_data, axis=0)
        mean_diff = np.mean(np.abs(orig_mean - gen_mean))
        std_diff = np.mean(np.abs(orig_std - gen_std))
        ks_statistics = []
        for i in range(min(original_data.shape[1], 10)):
            ks_stat, p_value = stats.ks_2samp(original_data[:, i], generated_data[:, i])
            ks_statistics.append(ks_stat)
        mean_ks = np.mean(ks_statistics)
        return mean_diff, std_diff, mean_ks

    def visualize_data_quality(self, original_data, generated_data, save_path=''):
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 3, 1)
        combined_data = np.vstack([original_data, generated_data])
        pca = PCA(n_components=2)
        combined_pca = pca.fit_transform(combined_data)
        orig_pca = combined_pca[:len(original_data)]
        gen_pca = combined_pca[len(original_data):]
        plt.scatter(orig_pca[:, 0], orig_pca[:, 1], alpha=0.6, label='Origin Data', s=20)
        plt.scatter(gen_pca[:, 0], gen_pca[:, 1], alpha=0.6, label='Sampled data', s=20)
        plt.title('PCA Visualization')
        plt.legend()
        plt.grid(True)
        for i in range(min(3, original_data.shape[1])):
            plt.subplot(2, 3, i + 2)
            plt.hist(original_data[:, i], bins=30, alpha=0.7, label='Origin Data', density=True)
            plt.hist(generated_data[:, i], bins=30, alpha=0.7, label='Sampled data', density=True)
            plt.title(f'Comparison of feature {i} distribution')
            plt.legend()
            plt.grid(True)
        plt.subplot(2, 3, 5)
        sample_size = min(200, len(original_data), len(generated_data))
        orig_sample = original_data[:sample_size]
        gen_sample = generated_data[:sample_size]
        orig_distances = pairwise_distances(orig_sample)
        gen_distances = pairwise_distances(gen_sample)
        orig_nn_dist = [np.sort(row)[1] for row in orig_distances]
        gen_nn_dist = [np.sort(row)[1] for row in gen_distances]
        plt.hist(orig_nn_dist, bins=20, alpha=0.7, label='Original nearest neighbor distance', density=True)
        plt.hist(gen_nn_dist, bins=20, alpha=0.7, label='Sampled nearest neighbor distance', density=True)
        plt.title('Nearest neighbor distance distribution')
        plt.legend()
        plt.grid(True)
        plt.subplot(2, 3, 6)
        scaler = StandardScaler()
        original_data = scaler.fit_transform(original_data)
        generated_data = scaler.fit_transform(generated_data)
        all_original = original_data.flatten()
        all_generated = generated_data.flatten()
        plt.hist(all_original, bins=50, alpha=0.6, label='Origin Data', density=True, color='skyblue',
                 edgecolor='black')
        plt.hist(all_generated, bins=50, alpha=0.6, label='Sampled Data', density=True, color='orange',
                 edgecolor='black')
        plt.title('Histogram of All Features Combined')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def diversity_analysis(self, generated_data):
        distances = pairwise_distances(generated_data)
        upper_triangle = distances[np.triu_indices_from(distances, k=1)]
        mean_distance = np.mean(upper_triangle)
        std_distance = np.std(upper_triangle)
        min_distance = np.min(upper_triangle[upper_triangle > 0])
        duplicate_threshold = 1e-6
        return mean_distance, std_distance

class MinorityVAEOversampler:
    def __init__(self, latent_dim=64, hidden_dim=128, lr=5e-4, epochs=100,
                 batch_size=64, beta=1.0, device='cpu'):
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.beta = beta
        self.device = device
        self.vae = None
        self.scaler = None
        self.minority_class = None
        self.majority_class = None
        self.evaluator = VAEEvaluator()

    def _normalize_features(self, features):
        if self.scaler is None:
            self.scaler = MinMaxScaler(feature_range=(-1, 1))
            normalized = self.scaler.fit_transform(features)
        else:
            normalized = self.scaler.transform(features)
        return normalized

    def _vae_loss(self, recon_x, x, mu, logvar):
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.beta * kl_loss

    def fit_and_generate(self, features_path, labels_path):
        features = torch.load(features_path)
        labels = torch.load(labels_path)
        if isinstance(features, torch.Tensor):
            features_np = features.cpu().numpy()
        else:
            features_np = np.array(features)
        if isinstance(labels, torch.Tensor):
            labels_np = labels.cpu().numpy()
        else:
            labels_np = np.array(labels)
        label_counts = Counter(labels_np)
        sorted_classes = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
        self.majority_class, majority_count = sorted_classes[0]
        self.minority_class, minority_count = sorted_classes[1]
        needed_samples = majority_count - minority_count
        if needed_samples <= 0:
            return torch.empty(0, features_np.shape[1]), torch.empty(0, dtype=torch.long)
        minority_mask = labels_np == self.minority_class
        minority_features = features_np[minority_mask]
        minority_labels = labels_np[minority_mask]
        features_normalized = self._normalize_features(minority_features)
        X = torch.FloatTensor(features_normalized).to(self.device)
        y = torch.LongTensor(minority_labels).to(self.device)
        input_dim = X.shape[1]
        num_classes = len(np.unique(labels_np))
        self.vae = ConditionalVAE(input_dim, self.latent_dim,
                                  self.hidden_dim, num_classes).to(self.device)

        optimizer = optim.Adam(self.vae.parameters(), lr=self.lr)
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=min(self.batch_size, len(X)),
                                                 shuffle=True)
        self.vae.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                recon_x, mu, logvar = self.vae(batch_x, batch_y)
                loss = self._vae_loss(recon_x, batch_x, mu, logvar)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(dataloader)
            self.evaluator.training_losses.append(avg_loss)
            if (epoch + 1) % 20 == 0:
                print(f'Epoch [{epoch + 1}/{self.epochs}], Loss: {avg_loss:.4f}')
        recon_error, correlation = self.evaluator.evaluate_reconstruction_quality(
            self.vae, features_normalized, minority_labels, self.device
        )
        gen_labels = torch.LongTensor([self.minority_class] * needed_samples)
        generated_features = self.vae.sample(needed_samples, gen_labels, self.device)
        generated_features_np = generated_features.cpu().numpy()
        generated_features_denorm = self.scaler.inverse_transform(generated_features_np)
        mean_diff, std_diff, mean_ks = self.evaluator.compare_distributions(
            minority_features, generated_features_denorm
        )
        self.evaluator.visualize_data_quality(minority_features, generated_features_denorm)
        self.evaluator.diversity_analysis(generated_features_denorm)
        generated_features_tensor = torch.FloatTensor(generated_features_denorm)
        generated_labels_tensor = torch.LongTensor([self.minority_class] * needed_samples)
        return generated_features_tensor, generated_labels_tensor


def generate_minority_samples(features_path="x.pt", labels_path="label.pt",
                              output_features_path="generated_features.pt",
                              output_labels_path="generated_labels.pt",
                              latent_dim=32, epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    oversampler = MinorityVAEOversampler(
        latent_dim=64,
        hidden_dim=1024,
        epochs=epochs,
        batch_size=64,
        beta=0.3,
        device=device,
        lr = 5e-4
    )
    try:
        generated_features, generated_labels = oversampler.fit_and_generate(
            features_path, labels_path
        )
        if len(generated_features) > 0:
            torch.save(generated_features, output_features_path)
            torch.save(generated_labels, output_labels_path)
            return generated_features, generated_labels
        else:
            return None, None
    except Exception as e:
        return None, None