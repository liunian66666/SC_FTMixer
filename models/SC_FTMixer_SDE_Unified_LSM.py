"""
SC_FTMixer_SDE_Unified + optional Low-Rank Sensor Mixer.

Core structure:
    1. Instance normalization
    2. FFT over time
    3. Static spectrum: global prototype + gate * calendar residual
    4. Dynamic residual filtering
    5. iFFT
    6. Linear temporal decoder
    7. Optional low-rank sensor mixer
    8. De-normalization
"""

import torch
import torch.nn as nn


class LowRankSensorMixer(nn.Module):
    """Lightweight low-rank cross-sensor mixer.
    Input:  x [B, T, C]
    Output: x + sigmoid(alpha) * Up(Down(LN(x)))
    Params: ~ 2*C*rank + 2*C + 1
    """
    def __init__(self, enc_in, rank=8, alpha_init=-4.0, use_norm=True):
        super().__init__()
        self.enc_in = enc_in
        self.rank = rank
        self.use_norm = use_norm
        self.norm = nn.LayerNorm(enc_in) if use_norm else nn.Identity()
        self.down = nn.Linear(enc_in, rank, bias=False)
        self.up = nn.Linear(rank, enc_in, bias=False)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.xavier_uniform_(self.up.weight)

    def forward(self, x):
        residual = self.up(self.down(self.norm(x)))
        scale = torch.sigmoid(self.alpha)
        return x + scale * residual


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = getattr(configs, "task_name", "long_term_forecast")
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.freq_bins = self.seq_len // 2 + 1

        # SDE switches
        self.use_sde = int(getattr(configs, "use_sde", 1))
        self.use_global_sde = int(getattr(configs, "use_global_sde", 1))
        self.use_calendar_sde = int(getattr(configs, "use_calendar_sde", 1))
        self.use_dynamic_filter = int(getattr(configs, "use_dynamic_filter", 1))
        self.fix_calendar_gate = int(getattr(configs, "fix_calendar_gate", 0))

        # Phase settings
        self.sde_cycle_len = int(getattr(configs, "sde_cycle_len", 24))
        self.sde_phase_mode = getattr(configs, "sde_phase_mode", "hour")
        self.sde_slots_per_hour = int(getattr(configs, "sde_slots_per_hour", 1))

        # Static spectrum parameters
        self.global_spectrum = nn.Parameter(
            torch.zeros(1, self.enc_in, self.freq_bins))
        self.calendar_residual = nn.Parameter(
            torch.zeros(self.sde_cycle_len, self.enc_in, self.freq_bins))
        gate_init = float(getattr(configs, "sde_calendar_gate_init", 2.0))
        self.calendar_gate = nn.Parameter(
            torch.full((1, self.enc_in, 1), gate_init))
        nn.init.normal_(self.global_spectrum, mean=0.0, std=0.02)
        nn.init.normal_(self.calendar_residual, mean=0.0, std=0.02)

        # Dynamic frequency filter
        self.dynamic_filter = nn.Parameter(
            torch.zeros(1, self.enc_in, self.freq_bins))

        # Temporal decoder
        self.decoder = nn.Linear(self.seq_len, self.pred_len)

        # Optional low-rank sensor mixer
        self.use_sensor_mixer = int(getattr(configs, "use_sensor_mixer", 0))
        self.sensor_rank = int(getattr(configs, "sensor_rank", 8))
        self.sensor_alpha_init = float(getattr(configs, "sensor_alpha_init", -4.0))
        if self.use_sensor_mixer:
            self.sensor_mixer = LowRankSensorMixer(
                enc_in=self.enc_in, rank=self.sensor_rank,
                alpha_init=self.sensor_alpha_init, use_norm=True)
        else:
            self.sensor_mixer = nn.Identity()

    def _get_phase(self, x_mark_enc, batch_size, device):
        if x_mark_enc is None or not torch.is_tensor(x_mark_enc) or x_mark_enc.dim() != 3:
            return torch.zeros(batch_size, dtype=torch.long, device=device)
        raw = x_mark_enc[:, -1, -1].to(device)
        phase = torch.round(raw).long() % self.sde_cycle_len
        return phase

    def _build_static_spectrum(self, spectrum, phase):
        batch_size = spectrum.shape[0]
        if not self.use_sde:
            return torch.zeros_like(spectrum.real)
        static = torch.zeros_like(spectrum.real)
        if self.use_global_sde:
            static = static + self.global_spectrum.expand(batch_size, -1, -1)
        if self.use_calendar_sde:
            calendar = self.calendar_residual[phase]
            if self.fix_calendar_gate:
                gate = torch.sigmoid(self.calendar_gate.detach())
            else:
                gate = torch.sigmoid(self.calendar_gate)
            static = static + gate * calendar
        return static

    def _reconstruct_spectrum(self, spectrum, phase):
        static = self._build_static_spectrum(spectrum, phase)
        dynamic = torch.complex(spectrum.real - static, spectrum.imag)
        if self.use_dynamic_filter:
            gain = 1.0 + torch.tanh(self.dynamic_filter)
            dynamic = dynamic * gain
        reconstructed = torch.complex(dynamic.real + static, dynamic.imag)
        return reconstructed

    def forecast(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        batch_size, seq_len, channels = x_enc.shape
        device = x_enc.device

        means = x_enc.mean(dim=1, keepdim=True).detach()
        x = x_enc - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x = x / stdev

        x_cf = x.permute(0, 2, 1).contiguous()
        spectrum = torch.fft.rfft(x_cf, dim=-1, norm="ortho")

        phase = self._get_phase(x_mark_enc, batch_size, device)
        reconstructed = self._reconstruct_spectrum(spectrum, phase)

        x_rec = torch.fft.irfft(reconstructed, n=self.seq_len, dim=-1, norm="ortho")
        dec_out = self.decoder(x_rec)
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        if self.use_sensor_mixer:
            dec_out = self.sensor_mixer(dec_out)

        dec_out = dec_out * stdev[:, 0:1, :] + means[:, 0:1, :]
        return dec_out

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name in ["long_term_forecast", "short_term_forecast", "forecast"]:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        raise NotImplementedError(f"Task {self.task_name} not implemented.")

    def sensor_mixer_param_count(self):
        if not self.use_sensor_mixer:
            return 0
        return sum(p.numel() for p in self.sensor_mixer.parameters())
