"""
Unified Global-Calendar Residual Static-Dynamic Embedding
with Optional Low-Rank Sensor Mixer.

One architecture is used for every dataset:

    static spectrum = global prototype
                    + bounded gate * calendar residual prototype

The global path is always present. Calendar conditioning is a residual that
can be suppressed automatically when it is unhelpful. Dataset frequency only
defines how timestamps map to phase indices; it does not change the network.

Optional extension:
    Low-Rank Sensor Mixer is only used when use_sensor_mixer=1.
    When use_sensor_mixer=0, the model should be equivalent to the original
    SC_FTMixer_SDE_Unified.
"""

import torch
import torch.nn as nn


def _get(configs, name, default):
    return getattr(configs, name, default)


class LowRankSensorMixer(nn.Module):
    """
    Conservative low-rank cross-sensor mixer.

    Input:
        x: [B, T, C]

    Output:
        x + scale * Up(Down(LN(x)))

    Parameter count:
        approximately 2 * C * rank + 2 * C + 1

    Important design:
        - up.weight is initialized to zero.
        - scale is bounded by max_scale * sigmoid(alpha).
        - Therefore, at initialization, the residual branch is exactly zero.
          This makes the module start from the original model behavior.
    """
    def __init__(self, enc_in, rank=4, alpha_init=-6.0, max_scale=0.1):
        super().__init__()

        self.enc_in = enc_in
        self.rank = rank
        self.max_scale = float(max_scale)

        self.norm = nn.LayerNorm(enc_in)
        self.down = nn.Linear(enc_in, rank, bias=False)
        self.up = nn.Linear(rank, enc_in, bias=False)

        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        # x: [B, T, C]
        residual = self.up(self.down(self.norm(x)))
        scale = self.max_scale * torch.sigmoid(self.alpha)
        return x + scale * residual


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in

        self.hidden = int(_get(configs, "sde_hidden", 192))
        self.cycle_len = int(_get(configs, "sde_cycle_len", 24))
        self.freq = str(_get(configs, "freq", "h")).lower()
        self.phase_mode = str(_get(configs, "sde_phase_mode", "auto")).lower()
        self.slots_per_hour = int(_get(configs, "sde_slots_per_hour", 1))

        self.use_sde = bool(int(_get(configs, "use_sde", 1)))
        self.use_global_sde = bool(int(_get(configs, "use_global_sde", 1)))
        self.use_calendar_sde = bool(int(_get(configs, "use_calendar_sde", 1)))
        self.use_dynamic_filter = bool(int(_get(configs, "use_dynamic_filter", 1)))
        self.fix_calendar_gate = bool(int(_get(configs, "fix_calendar_gate", 0)))

        self.output_attention = bool(_get(configs, "output_attention", False))

        freq_bins = self.seq_len // 2 + 1

        # Dataset-independent base prior.
        self.global_spectrum = nn.Parameter(
            torch.zeros(1, self.enc_in, freq_bins)
        )

        # Calendar is only a residual correction around the global prior.
        self.calendar_residual = nn.Parameter(
            torch.zeros(self.cycle_len, self.enc_in, freq_bins)
        )

        # Keep original default as -2.0.
        # In experiments you can still pass --sde_calendar_gate_init 2.0.
        gate_init = float(_get(configs, "sde_calendar_gate_init", -2.0))
        self.calendar_gate = nn.Parameter(
            torch.full((1, self.enc_in, 1), gate_init)
        )

        # Original dynamic filter.
        self.dynamic_filter = nn.Parameter(
            0.02 * torch.randn(1, self.seq_len)
        )

        # Original temporal decoder.
        self.decoder = nn.Sequential(
            nn.Linear(self.seq_len, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.pred_len),
        )

        # ------------------------------------------------------------
        # Optional Low-Rank Sensor Mixer
        # ------------------------------------------------------------
        # Important:
        #   When use_sensor_mixer=0, no extra module is created.
        #   This helps preserve the original model behavior.
        self.use_sensor_mixer = bool(int(_get(configs, "use_sensor_mixer", 0)))

        if self.use_sensor_mixer:
            self.sensor_rank = int(_get(configs, "sensor_rank", 4))
            self.sensor_alpha_init = float(_get(configs, "sensor_alpha_init", -6.0))
            self.sensor_max_scale = float(_get(configs, "sensor_max_scale", 0.1))

            self.sensor_mixer = LowRankSensorMixer(
                enc_in=self.enc_in,
                rank=self.sensor_rank,
                alpha_init=self.sensor_alpha_init,
                max_scale=self.sensor_max_scale,
            )

    def _mode(self):
        if self.phase_mode != "auto":
            return self.phase_mode

        if self.freq.startswith(("d", "b")):
            return "weekday"

        if self.freq.startswith(("t", "min", "s")):
            return "day_slot"

        return "hour_week"

    def _forecast_mark(self, x_mark_enc, x_mark_dec):
        if x_mark_dec is not None and x_mark_dec.ndim == 3:
            return x_mark_dec[:, -self.pred_len, :]

        if x_mark_enc is not None and x_mark_enc.ndim == 3:
            return x_mark_enc[:, -1, :]

        return None

    def _phase_index(
        self,
        x_mark_enc,
        x_mark_dec,
        position_index,
        batch_size,
        device,
    ):
        mode = self._mode()

        if mode == "position":
            if position_index is None:
                raise ValueError(
                    "sde_phase_mode=position requires x_enc_long phase indices"
                )

            return position_index.reshape(-1).long().to(device) % self.cycle_len

        mark = self._forecast_mark(x_mark_enc, x_mark_dec)

        if mark is None:
            return torch.zeros(batch_size, dtype=torch.long, device=device)

        if mode == "weekday":
            phase = torch.round((mark[:, 0] + 0.5) * 6.0)

        elif mode == "day_slot":
            minute = torch.round((mark[:, 0] + 0.5) * 59.0)
            hour = torch.round((mark[:, 1] + 0.5) * 23.0)
            minute_slot = torch.floor(
                minute * self.slots_per_hour / 60.0
            )
            phase = hour * self.slots_per_hour + minute_slot

        elif mode == "hour":
            phase = torch.round((mark[:, 0] + 0.5) * 23.0)

        elif mode == "hour_week":
            hour = torch.round((mark[:, 0] + 0.5) * 23.0)
            weekday = torch.round((mark[:, 1] + 0.5) * 6.0)
            phase = weekday * 24 + hour

        else:
            raise ValueError(f"Unsupported sde_phase_mode: {mode}")

        return phase.long().remainder(self.cycle_len)

    def forecast(
        self,
        x_enc,
        x_mark_enc=None,
        x_mark_dec=None,
        position_index=None,
    ):
        # ------------------------------------------------------------
        # Instance normalization
        # ------------------------------------------------------------
        mean = x_enc.mean(dim=1, keepdim=True)
        var = x_enc.var(dim=1, keepdim=True)
        std = torch.sqrt(var + 1e-5)

        # [B, L, C] -> [B, C, L]
        x = ((x_enc - mean) / std).transpose(1, 2)

        # ------------------------------------------------------------
        # FFT
        # ------------------------------------------------------------
        spectrum = torch.fft.rfft(x, dim=-1, norm="ortho")

        phase = self._phase_index(
            x_mark_enc,
            x_mark_dec,
            position_index,
            x_enc.size(0),
            x_enc.device,
        )

        # ------------------------------------------------------------
        # Static spectrum
        # ------------------------------------------------------------
        if self.use_sde:
            if self.fix_calendar_gate:
                gate = torch.ones_like(self.calendar_gate)
            else:
                gate = torch.sigmoid(self.calendar_gate)

            static = torch.zeros_like(spectrum.real)

            if self.use_global_sde:
                static = static + self.global_spectrum

            if self.use_calendar_sde:
                static = static + gate * self.calendar_residual[phase]

        else:
            static = torch.zeros_like(spectrum.real)

        # ------------------------------------------------------------
        # Dynamic residual filtering
        # ------------------------------------------------------------
        dynamic = torch.complex(
            spectrum.real - static,
            spectrum.imag,
        )

        if self.use_dynamic_filter:
            filt = torch.fft.rfft(
                self.dynamic_filter,
                dim=-1,
                norm="ortho",
            )
            filtered = dynamic * filt

            reconstructed = torch.complex(
                filtered.real + static,
                filtered.imag,
            )

        else:
            # Static + dynamic exactly reconstructs the original spectrum.
            reconstructed = torch.complex(
                dynamic.real + static,
                dynamic.imag,
            )

        # ------------------------------------------------------------
        # iFFT
        # ------------------------------------------------------------
        z = torch.fft.irfft(
            reconstructed,
            n=self.seq_len,
            dim=-1,
            norm="ortho",
        )

        # ------------------------------------------------------------
        # Temporal decoder
        # [B, C, L] -> [B, C, pred_len] -> [B, pred_len, C]
        # ------------------------------------------------------------
        y = self.decoder(z).transpose(1, 2)

        # ------------------------------------------------------------
        # Optional sensor mixer
        # Apply in normalized space, before de-normalization.
        # ------------------------------------------------------------
        if self.use_sensor_mixer:
            y = self.sensor_mixer(y)

        # ------------------------------------------------------------
        # De-normalization
        # ------------------------------------------------------------
        y = y * std + mean

        return y, phase, torch.sigmoid(self.calendar_gate)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
        mask=None,
        **kwargs,
    ):
        y, phase, gate = self.forecast(
            x_enc,
            x_mark_enc,
            x_mark_dec,
            kwargs.get("x_enc_long"),
        )

        if self.output_attention:
            return y, {
                "phase": phase,
                "calendar_gate": gate,
            }

        return y

