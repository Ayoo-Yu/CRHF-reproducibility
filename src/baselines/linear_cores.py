"""Study-specific 96-step linear backbones. See THIRD_PARTY.md for attribution."""
import torch
from torch import nn
from torch.nn import functional as F


class MixingCore(nn.Module):
    """Combined period-grid and low-frequency forecasts, with a four-step readout."""
    def __init__(self, frequencies):
        super().__init__()
        # 96 observations form 24 periods of four. Pad 24 periods to a 5x5 grid.
        self.row_projection = nn.Linear(5, 1, bias=False)
        self.column_projection = nn.Linear(5, 1, bias=False)
        self.smoothing = nn.Conv1d(1, 1, kernel_size=5, padding=2, bias=False)
        self.frequency_projection = nn.Linear(frequencies, 2, bias=False).to(torch.cfloat)
        self.frequency_readout = nn.Linear(2, 1, bias=False).to(torch.cfloat)
        self.frequencies = frequencies

    def forward(self, sequence):
        batch, _, channels = sequence.shape
        center = sequence.mean(dim=1, keepdim=True)
        anomalies = (sequence - center).transpose(1, 2)
        smoothed = self.smoothing(anomalies.reshape(-1, 1, 96)).reshape(batch, channels, 96)
        phases = (smoothed + anomalies).reshape(batch, channels, 24, 4).transpose(2, 3)
        padded = F.pad(phases, (0, 1))
        grid = padded.reshape(batch, channels, 4, 5, 5)
        rows = self.row_projection(grid).transpose(-1, -2)
        temporal = self.column_projection(rows).reshape(batch, channels, 4).transpose(1, 2)
        spectrum = torch.fft.fft(phases, dim=-1)[..., :self.frequencies]
        reduced = self.frequency_readout(self.frequency_projection(spectrum))
        spectral = torch.fft.ifft(reduced, dim=-1).real.squeeze(-1).transpose(1, 2)
        return temporal * 0.5 + center + spectral * 0.5


class DecompositionCore(nn.Module):
    """Separate 25-slot moving-average and seasonal projections with channel mixing."""
    def __init__(self):
        super().__init__()
        self.seasonal = nn.Linear(96, 1)
        self.trend = nn.Linear(96, 1)
        # Preserve random-number consumption of the registered, unused decoder.
        self.decoder = nn.Linear(96, 1)
        self.readout = nn.Linear(6, 1)
        for projection in (self.seasonal, self.trend, self.readout):
            nn.init.constant_(projection.weight, 1 / 96)

    def forward(self, sequence):
        signal = sequence.transpose(1, 2)
        padded = F.pad(signal, (12, 12), mode="replicate")
        slow = F.avg_pool1d(padded, kernel_size=25, stride=1)
        fast = signal - slow
        per_channel = self.seasonal(fast) + self.trend(slow)
        return self.readout(per_channel.transpose(1, 2))
