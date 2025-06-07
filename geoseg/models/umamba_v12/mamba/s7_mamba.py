import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat

# from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn, mamba_inner_fn_no_out_proj

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class Mamba(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=4,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_bias=True,
            bias=False,
            use_fast_path=True,  # Fused kernel options
            layer_idx=None,
            device=None,
            dtype=None,
            bimamba_type="none",
            if_devide_out=True,
            init_layer_scale=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.bimamba_type = bimamba_type
        self.if_devide_out = if_devide_out
        self.init_layer_scale = init_layer_scale

        if init_layer_scale is not None:
            self.gamma = nn.Parameter(init_layer_scale * torch.ones((d_model)), requires_grad=True)

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        # bimamba
        A_b = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_b_log = torch.log(A_b)  # Keep A_b_log in fp32
        self.A_b_log = nn.Parameter(A_b_log)
        self.A_b_log._no_weight_decay = True

        self.conv1d_b = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.x_proj_b = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        self.D_b = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D_b._no_weight_decay = True

        # random mamba
        A_r = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_r_log = torch.log(A_r)  # Keep A_b_log in fp32
        self.A_r_log = nn.Parameter(A_r_log)
        self.A_r_log._no_weight_decay = True

        self.conv1d_r = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.x_proj_r = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj_r = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        self.D_r = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D_r._no_weight_decay = True

        # # bidirectional
        # if bimamba_type == "v1":
        #     # 控制门函数
        #     # gate_out_dim = 2
        #     # self.gate_linear = nn.Sequential(
        #     #     nn.Linear(self.expand*gate_out_dim*d_model, gate_out_dim, bias=False),
        #     #     nn.Softmax(dim=-1)
        #     # )
        #     self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        # elif bimamba_type == "v2":
        #     # 控制门函数
        #     # gate_out_dim = 3
        #     # self.gate_linear = nn.Sequential(
        #     #     nn.Linear(self.expand*gate_out_dim*d_model, gate_out_dim, bias=False),
        #     #     nn.Softmax(dim=-1)
        #     # )
        #     self.weights = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.act = nn.SiLU(inplace=True)

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if self.use_fast_path and causal_conv1d_fn is not None and inference_params is None:  # Doesn't support outputting the states
            if self.bimamba_type == "v1":
                A_b = -torch.exp(self.A_b_log.float())
                out = mamba_inner_fn_no_out_proj(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )
                out_b = mamba_inner_fn_no_out_proj(
                    xz.flip([-1]),
                    self.conv1d_b.weight,
                    self.conv1d_b.bias,
                    self.x_proj_b.weight,
                    self.dt_proj_b.weight,
                    A_b,
                    None,
                    None,
                    self.D_b.float(),
                    delta_bias=self.dt_proj_b.bias.float(),
                    delta_softplus=True,
                )
                # # F.linear(rearrange(out_z, "b d l -> b l d"), out_proj_weight, out_proj_bias)
                # mean_f = torch.mean(out, dim=-1)
                # mean_b = torch.mean(out_b.flip([-1]), dim=-1)
                # gate_fb = torch.cat([mean_f, mean_b], dim=1)
                # gate_fb = self.gate_linear(gate_fb)
                # gate_f = gate_fb.unsqueeze(1)[:,:,0:1]
                # gate_b = gate_fb.unsqueeze(1)[:,:,1:2]

                # if not self.if_devide_out:
                #     out = F.linear(rearrange(gate_f*(out) + gate_b*(out_b.flip([-1])), "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias)
                # else:
                #     out = F.linear(rearrange(gate_f*(out) + gate_b*(out_b.flip([-1])), "b d l -> b l d") / 2, self.out_proj.weight, self.out_proj.bias)
                if not self.if_devide_out:
                    out = F.linear(rearrange(out + out_b.flip([-1]), "b d l -> b l d"), self.out_proj.weight,
                                   self.out_proj.bias)
                else:
                    out = F.linear(rearrange(out + out_b.flip([-1]), "b d l -> b l d") / 2, self.out_proj.weight,
                                   self.out_proj.bias)
            # elif self.bimamba_type == 'v2':
            #     A_b = -torch.exp(self.A_b_log.float())
            #     A_r = -torch.exp(self.A_r_log.float())
            #     out = mamba_inner_fn_no_out_proj(
            #         xz,
            #         self.conv1d.weight,
            #         self.conv1d.bias,
            #         self.x_proj.weight,
            #         self.dt_proj.weight,
            #         A,
            #         None,  # input-dependent B
            #         None,  # input-dependent C
            #         self.D.float(),
            #         delta_bias=self.dt_proj.bias.float(),
            #         delta_softplus=True,
            #     )
            #     out_b = mamba_inner_fn_no_out_proj(
            #         xz.flip([-1]),
            #         self.conv1d_b.weight,
            #         self.conv1d_b.bias,
            #         self.x_proj_b.weight,
            #         self.dt_proj_b.weight,
            #         A_b,
            #         None,
            #         None,
            #         self.D_b.float(),
            #         delta_bias=self.dt_proj_b.bias.float(),
            #         delta_softplus=True,
            #     )
            #     idx = torch.randperm(xz.size(-1)).to('cuda:0')
            #     out_r = mamba_inner_fn_no_out_proj(
            #         xz.index_select(dim=-1, index=idx),
            #         self.conv1d_r.weight,
            #         self.conv1d_r.bias,
            #         self.x_proj_r.weight,
            #         self.dt_proj_r.weight,
            #         A_r,
            #         None,  # input-dependent B
            #         None,  # input-dependent C
            #         self.D_r.float(),
            #         delta_bias=self.dt_proj_r.bias.float(),
            #         delta_softplus=True,
            #     )
            #     # F.linear(rearrange(out_z, "b d l -> b l d"), out_proj_weight, out_proj_bias)
            #     idx_inverse = torch.argsort(idx)
            #
            #     # mean_f = torch.mean(out, dim=-1)
            #     # mean_b = torch.mean(out_b.flip([-1]), dim=-1)
            #     # mean_r = torch.mean(out_r.index_select(dim=-1, index=idx_inverse), dim=-1)
            #     # gate_fbr = torch.cat([mean_f, mean_b, mean_r], dim=1)
            #     # gate_fbr = self.gate_linear(gate_fbr)
            #     # gate_f = gate_fbr.unsqueeze(1)[:,:,0:1]
            #     # gate_b = gate_fbr.unsqueeze(1)[:,:,1:2]
            #     # gate_r = gate_fbr.unsqueeze(1)[:,:,2:3]
            #
            #     # if not self.if_devide_out:
            #     #     out = F.linear(rearrange(gate_f*(out) + gate_b*(out_b.flip([-1])) + gate_r*(out_r.index_select(dim=-1, index=idx_inverse)), "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias)
            #     #     # out = F.linear(rearrange(gate_f*(out) + gate_b*(out_b.flip([-1])) + gate_r*(out_r), "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias)
            #     # else:
            #     #     # out = F.linear(rearrange(gate_f*(out) + gate_b*(out_b.flip([-1])) + gate_r*(out_r), "b d l -> b l d") / 3, self.out_proj.weight, self.out_proj.bias)
            #     #     out = F.linear(rearrange(gate_f*(out) + gate_b*(out_b.flip([-1])) + gate_r*(out_r.index_select(dim=-1, index=idx_inverse)), "b d l -> b l d") / 3, self.out_proj.weight, self.out_proj.bias)
            #     if not self.if_devide_out:
            #         out = F.linear(rearrange(out + out_b.flip([-1]) + out_r.index_select(dim=-1, index=idx_inverse),
            #                                  "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias)
            #         # out = F.linear(rearrange(gate_f*(out) + gate_b*(out_b.flip([-1])) + gate_r*(out_r), "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias)
            #     else:
            #         # out = F.linear(rearrange(gate_f*(out) + gate_b*(out_b.flip([-1])) + gate_r*(out_r), "b d l -> b l d") / 3, self.out_proj.weight, self.out_proj.bias)
            #         out = F.linear(rearrange(out + out_b.flip([-1]) + out_r.index_select(dim=-1, index=idx_inverse),
            #                                  "b d l -> b l d") / 3, self.out_proj.weight, self.out_proj.bias)
            else:
                out = mamba_inner_fn(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )
        else:
            x, z = xz.chunk(2, dim=1)
            # Compute short convolution
            if conv_state is not None:
                # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state
