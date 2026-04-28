# Cursor 명령서: DOURI Curved Centerline Bridge 옵션 구현

Repository:

```text
https://github.com/SChoish/douri.git
```

## 목표

DOURI의 forward state-space bridge에 **optional curved centerline bridge mode**를 구현한다.

이 기능은 반드시 옵션이어야 하며, 기본값은 꺼져 있어야 한다. 옵션이 꺼져 있을 때는 기존 linear-SDE bridge의 동작이 바뀌면 안 된다.

DOURI의 기존 구조는 다음 흐름을 유지한다.

```text
subgoal prediction
    -> forward state-space bridge rollout
    -> inverse dynamics
    -> action chunks
    -> partial critic scoring
    -> Q-Boltzmann proposal
    -> SPI actor extraction
```

이번 변경은 bridge trajectory proposal만 더 표현력 있게 만드는 것이다. inverse dynamics, chunk critic, Q-Boltzmann proposal, SPI actor loss는 shape compatibility 문제가 없는 한 수정하지 않는다.

---

## 1. 배경 개념

기존 DOURI bridge는 displacement coordinate에서 정의된다.

```math
r_i = s_i - s_0, \qquad \delta = s_K - s_0.
```

기존 passive linear-SDE bridge는 대략 다음 형태다.

```math
r_{i+1} = A_i r_i + \eta_i,
\qquad
A_i = e^{\theta_i},
\qquad
\eta_i \sim \mathcal{N}(0,q_i^2 I).
```

terminal condition을 걸면 analytic one-step bridge teacher가 다음처럼 계산된다.

```math
\mu_i^{\star,\gamma}
=
s_0
+
e^{\theta_i}(s_i-s_0)
+
\kappa_i^\gamma
\left[
(s_K-s_0)-F_{i:K}(s_i-s_0)
\right].
```

기존 exact-residual mode는 이 teacher 위에 variance-scaled residual을 얹는다.

```math
\mu_i^{\mathrm{res}}
=
\mu_i^{\star,\gamma}
+
\alpha_{\mathrm{res}}\sqrt{C_i^\gamma}
\,r_\psi(s_i,s_0,s_K,i).
```

이 구조의 장점은 closed-form bridge posterior와 endpoint control이다. 단점은 bridge marginal mean이 대략 직선 interpolation에 가까워서, maze나 constrained manipulation에서 path가 벽을 뚫거나 infeasible state proposal을 만들 수 있다는 점이다.

Curved centerline bridge는 기존 analytic bridge를 버리지 않고, **endpoint-preserving learned centerline 주변의 residual coordinate에서 linear bridge를 수행**한다.

---

## 2. 핵심 아이디어

기존 평균 경로가 대략

```math
\mathbb{E}[s_i \mid s_0,s_K]
\approx
(1-\beta_i)s_0+\beta_i s_K
```

꼴이었다면, 이제 neural network가 중심 곡선을 예측한다.

```math
c_\psi(i;s_0,s_K,g)
=
(1-\beta_i)s_0
+
\beta_i s_K
+
\beta_i(1-\beta_i)
 h_\psi(s_0,s_K,g,i/K).
```

여기서 중요한 것은 envelope term이다.

```math
\beta_i(1-\beta_i).
```

만약

```math
\beta_0=0, \qquad \beta_K=1,
```

이면 자동으로

```math
c_\psi(0)=s_0, \qquad c_\psi(K)=s_K
```

가 된다. 즉 neural network가 중간 경로만 휘게 만들고 endpoint는 고정된다.

그다음 bridge를 raw state가 아니라 centerline 주변 residual coordinate에서 정의한다.

```math
z_i = s_i - c_i,
\qquad
c_i := c_\psi(i;s_0,s_K,g).
```

그러면 endpoint에서는

```math
z_0 = 0,
\qquad
z_K = 0.
```

따라서 residual coordinate에서는 zero-to-zero bridge가 된다.

```math
z_0 \to z_K = 0.
```

기존 analytic bridge를 z-space에 적용한다.

```math
\mu_{z,i}^{\star,\gamma}
=
A_i z_i
+
\kappa_i^\gamma (0-F_{i:K}z_i).
```

state coordinate로 돌아오면

```math
\mu_{s,i}^{\star,\gamma}
=
c_{i+1}
+
\mu_{z,i}^{\star,\gamma}.
```

그리고 exact residual mode를 유지한다.

```math
\mu_i
=
c_{i+1}
+
\mu_{z,i}^{\star,\gamma}
+
\alpha_{\mathrm{res}}\rho_i
r_\psi(s_i,s_0,s_K,g,i).
```

여기서 `rho_i`는 기존 variance scaling을 사용한다. hard endpoint safety를 위해 가능하면 hard-bridge variance 기반을 선호한다.

```math
\rho_i = \sqrt{C_i^\infty}
```

또는 기존 구현과 일관되게

```math
\rho_i = \sqrt{C_i^\gamma}.
```

---

## 3. 중요한 구현 원칙

### 3.1 Centerline은 현재 sampled state `s_i`에 의존하면 안 된다

Centerline은 반드시 다음에만 의존한다.

```text
s0, sK, goal g, time i/K
```

즉,

```math
c_i = c_\psi(i;s_0,s_K,g)
```

이어야 한다.

다음처럼 만들면 안 된다.

```math
c_i = c_\psi(s_i,s_0,s_K,g,i).
```

이유: `c_i`가 `s_i`에 의존하면

```math
z_i = s_i - c_\psi(s_i,\cdot)
```

가 nonlinear state-dependent transform이 되고, 기존 closed-form Gaussian bridge 해석이 깨진다.

### 3.2 Residual network는 `s_i`에 의존해도 된다

Residual은 local correction이므로 다음처럼 둘 수 있다.

```math
r_\psi(s_i,s_0,s_K,g,i).
```

### 3.3 옵션이 꺼져 있으면 기존 구현과 완전히 같아야 한다

`use_curved_centerline=False`인 경우 기존 bridge teacher, residual, rollout이 그대로 실행되어야 한다.

---

## 4. Repository에서 찾아야 할 코드

먼저 repo에서 관련 파일을 찾는다.

```bash
rg "mu_star|kappa|bridge|exact|residual|rollout|S0|F_|theta|gamma|C_i|path_loss|roll_loss" .
rg "inverse dynamics|invdyn|q_boltzmann|SPI|subgoal" .
rg "DOURI|douri|bridge_model|dynamics" .
```

찾아야 하는 항목:

- schedule computation: `theta_i`, `g2_i`, `q2_i`
- interval quantities: `F_{i:j}`, `S_{i:j}`, `kappa_i`, `C_i`
- existing exact teacher `mu_star_i`
- existing exact residual mode
- bridge rollout / sampling function
- path loss and rollout loss
- config/dataclass/argparse flags

Repo가 JAX/Flax 기반이면 JAX/Flax로 구현한다. PyTorch 기반이면 같은 로직을 PyTorch로 옮긴다. 새 framework를 도입하지 않는다.

---

## 5. Config 옵션 추가

기본값은 기존 동작 보존을 위해 모두 conservative하게 설정한다.

```python
use_curved_centerline: bool = False
centerline_hidden_dims: tuple = (256, 256)
centerline_scale: float = 1.0
centerline_zero_init: bool = True
centerline_beta_type: str = "linear"  # choices: "linear", "hard_bridge"
centerline_use_goal: bool = True
centerline_amp_coef: float = 1e-4
centerline_smooth_coef: float = 1e-3
centerline_residual_use_hard_variance: bool = True
centerline_apply_to_state_dims: Optional[list[int]] = None
```

구현 방식:

- argparse를 쓰면 argument로 추가한다.
- `ml_collections.ConfigDict`, dataclass, yaml config를 쓰면 해당 config에 추가한다.

동작:

- `use_curved_centerline=False`: 기존 bridge output과 동일해야 한다.
- `use_curved_centerline=True`: 기존 state-space bridge teacher 대신 curved centerline teacher를 사용한다.
- 기존 exact-residual mode는 그대로 작동해야 한다.

---

## 6. Centerline network 구현

### 6.1 입력

```text
s0:   [B, state_dim]
sK:   [B, state_dim]
goal: [B, goal_dim] or None
tau:  [B, 1], tau = i / K
beta: [B, 1]
```

추천 input concat:

```text
[s0, sK, sK - s0, goal, tau]
```

### 6.2 출력

```text
h: [B, state_dim]
```

### 6.3 centerline 계산

```math
c_i = (1-\beta_i)s_0 + \beta_i s_K
      + \beta_i(1-\beta_i) \cdot \text{centerline_scale} \cdot h.
```

### 6.4 JAX/Flax 예시

```python
import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence


class CurvedCenterline(nn.Module):
    state_dim: int
    hidden_dims: Sequence[int] = (256, 256)
    centerline_scale: float = 1.0
    zero_init: bool = True
    use_goal: bool = True

    @nn.compact
    def __call__(self, s0, sK, goal, tau, beta, mask=None):
        # s0, sK: [B, state_dim]
        # goal: [B, goal_dim] or None
        # tau, beta: [B, 1]

        inputs = [s0, sK, sK - s0, tau]
        if self.use_goal and goal is not None:
            inputs.append(goal)

        x = jnp.concatenate(inputs, axis=-1)

        for hdim in self.hidden_dims:
            x = nn.Dense(hdim)(x)
            x = nn.gelu(x)

        if self.zero_init:
            kernel_init = nn.initializers.zeros
            bias_init = nn.initializers.zeros
        else:
            kernel_init = nn.initializers.xavier_uniform()
            bias_init = nn.initializers.zeros

        h = nn.Dense(
            self.state_dim,
            kernel_init=kernel_init,
            bias_init=bias_init,
        )(x)

        if mask is not None:
            h = h * mask

        envelope = beta * (1.0 - beta)
        linear = (1.0 - beta) * s0 + beta * sK
        c = linear + envelope * self.centerline_scale * h
        return c, h
```

### 6.5 PyTorch 예시

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class CurvedCenterline(nn.Module):
    def __init__(
        self,
        state_dim: int,
        goal_dim: int = 0,
        hidden_dims=(256, 256),
        centerline_scale: float = 1.0,
        zero_init: bool = True,
        use_goal: bool = True,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.centerline_scale = centerline_scale
        self.use_goal = use_goal

        input_dim = state_dim * 3 + 1
        if use_goal:
            input_dim += goal_dim

        layers = []
        prev = input_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(prev, hdim))
            layers.append(nn.GELU())
            prev = hdim
        self.trunk = nn.Sequential(*layers)
        self.out = nn.Linear(prev, state_dim)

        if zero_init:
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

    def forward(self, s0, sK, goal, tau, beta, mask=None):
        inputs = [s0, sK, sK - s0, tau]
        if self.use_goal and goal is not None:
            inputs.append(goal)
        x = torch.cat(inputs, dim=-1)

        h = self.out(self.trunk(x))
        if mask is not None:
            h = h * mask

        envelope = beta * (1.0 - beta)
        linear = (1.0 - beta) * s0 + beta * sK
        c = linear + envelope * self.centerline_scale * h
        return c, h
```

---

## 7. Beta schedule 구현

Centerline beta는 반드시 endpoint를 보존해야 한다.

```math
\beta_0=0,\qquad \beta_K=1.
```

가장 단순한 기본값은 linear beta다.

```math
\beta_i = i/K.
```

hard bridge progress를 쓰고 싶으면 다음을 사용한다.

```math
\beta_i = \frac{S_{0:i}F_{i:K}}{S_{0:K}}.
```

finite-gamma soft beta는 `beta_K=1`을 만족하지 않을 수 있으므로 centerline endpoint interpolation에는 쓰지 않는다.

예시:

```python
def compute_centerline_beta(i, K, quantities=None, beta_type="linear"):
    if beta_type == "linear":
        return i / K
    elif beta_type == "hard_bridge":
        # beta_i = S_{0:i} * F_{i:K} / S_{0:K}
        return quantities.S_0i[i] * quantities.F_iK[i] / (quantities.S_0K + 1e-8)
    else:
        raise ValueError(f"Unknown centerline_beta_type: {beta_type}")
```

---

## 8. Curved bridge teacher 구현

기존 teacher가 다음을 계산했다면,

```python
r_i = s_i - s0
mu_star = s0 + A_i * r_i + kappa_i * (delta - F_iK * r_i)
```

curved mode에서는 다음을 계산한다.

```python
c_i = centerline(i)
c_next = centerline(i + 1)
z_i = s_i - c_i
mu_z_star = A_i * z_i + kappa_i * (-F_iK * z_i)
mu_star_curved = c_next + mu_z_star
```

수식으로는

```math
\mu_{\mathrm{curved},i}^{\star,\gamma}
=
c_{i+1}
+
\left(A_i-\kappa_i^\gamma F_{i:K}\right)(s_i-c_i).
```

JAX-style reference:

```python
def curved_bridge_teacher(
    s_i,
    s0,
    sK,
    goal,
    i,
    K,
    bridge_coeffs,
    centerline_apply_fn,
    centerline_params,
    cfg,
    mask=None,
):
    beta_i = compute_centerline_beta(
        i=i,
        K=K,
        quantities=bridge_coeffs,
        beta_type=cfg.centerline_beta_type,
    )
    beta_next = compute_centerline_beta(
        i=i + 1,
        K=K,
        quantities=bridge_coeffs,
        beta_type=cfg.centerline_beta_type,
    )

    # Broadcast to [B, 1].
    B = s_i.shape[0]
    beta_i = jnp.full((B, 1), beta_i)
    beta_next = jnp.full((B, 1), beta_next)
    tau_i = jnp.full((B, 1), i / K)
    tau_next = jnp.full((B, 1), (i + 1) / K)

    c_i, h_i = centerline_apply_fn(
        centerline_params,
        s0,
        sK,
        goal,
        tau_i,
        beta_i,
        mask,
    )
    c_next, h_next = centerline_apply_fn(
        centerline_params,
        s0,
        sK,
        goal,
        tau_next,
        beta_next,
        mask,
    )

    z_i = s_i - c_i

    A_i = bridge_coeffs.A_i[i]
    F_iK = bridge_coeffs.F_iK[i]
    kappa_i = bridge_coeffs.kappa_i[i]

    A_i = jnp.asarray(A_i).reshape(1, 1)
    F_iK = jnp.asarray(F_iK).reshape(1, 1)
    kappa_i = jnp.asarray(kappa_i).reshape(1, 1)

    mu_z_star = A_i * z_i + kappa_i * (-F_iK * z_i)
    mu_star_curved = c_next + mu_z_star

    aux = {
        "c_i": c_i,
        "c_next": c_next,
        "h_i": h_i,
        "h_next": h_next,
        "z_i": z_i,
    }
    return mu_star_curved, aux
```

PyTorch에서는 `jnp`를 `torch`로 바꾸고 broadcasting만 맞춘다.

---

## 9. Existing exact-residual mode와 결합

기존 코드가 다음처럼 되어 있다면,

```python
mu_star = existing_bridge_teacher(...)
mu = mu_star + alpha_res * sqrt_C * residual_net(...)
```

다음처럼 바꾼다.

```python
if cfg.use_curved_centerline:
    mu_star, center_aux = curved_bridge_teacher(...)
else:
    mu_star, center_aux = existing_bridge_teacher(...), {}

if cfg.exact_residual:
    if cfg.centerline_residual_use_hard_variance and hasattr(bridge_coeffs, "C_hard_i"):
        rho = sqrt(max(bridge_coeffs.C_hard_i[i], 0.0))
    else:
        rho = sqrt(max(bridge_coeffs.C_i[i], 0.0))

    res = residual_net(s_i, s0, sK, goal, i)
    mu = mu_star + cfg.alpha_res * rho * res
else:
    mu = mu_star
```

JAX-style:

```python
if cfg.use_curved_centerline:
    mu_star, center_aux = curved_bridge_teacher(...)
else:
    mu_star, center_aux = existing_mu_star, {}

if cfg.exact_residual:
    if cfg.centerline_residual_use_hard_variance and hasattr(bridge_coeffs, "C_hard_i"):
        C_for_res = bridge_coeffs.C_hard_i[i]
    else:
        C_for_res = bridge_coeffs.C_i[i]

    rho = jnp.sqrt(jnp.maximum(C_for_res, 0.0)).reshape(1, 1)
    res = residual_net_apply_fn(residual_params, s_i, s0, sK, goal, i)
    mu = mu_star + cfg.alpha_res * rho * res
else:
    mu = mu_star
```

Important:

- hard bridge final step에서 `C_{K-1}=0`이면 residual이 자동으로 사라져야 한다.
- `mu_{K-1}=sK`가 되도록 test를 추가한다.

---

## 10. Training losses

기존 path loss와 rollout loss는 유지한다. 단, curved mode에서는 새 model mean을 사용한다.

### 10.1 Path loss

```math
L_{\mathrm{path}}
=
\mathbb{E}_i
\left[
\left\|
s_{t+i+1}^D
-
\mu_i(s_{t+i}^D,s_t^D,s_{t+K}^D,g)
\right\|_1
\right].
```

### 10.2 Rollout loss

```math
\hat{s}_t=s_t^D,
```

```math
\hat{s}_{t+i+1}
=
\mu_i(\hat{s}_{t+i},s_t^D,s_{t+K}^D,g),
```

```math
L_{\mathrm{roll}}
=
\frac{1}{K}
\sum_{i=1}^{K}
\left\|
\hat{s}_{t+i}-s_{t+i}^D
\right\|_1.
```

### 10.3 Centerline amplitude regularizer

Raw deformation `h_i`를 penalize한다. 전체 `c_i`가 아니라 `h_i`를 사용한다.

```math
L_{\mathrm{center\_amp}}
=
\mathbb{E}_i \|h_i\|_2^2.
```

### 10.4 Centerline smoothness regularizer

```math
L_{\mathrm{center\_smooth}}
=
\sum_{i=1}^{K-1}
\left\|
c_{i+1}-2c_i+c_{i-1}
\right\|_2^2.
```

Reference code:

```python
def centerline_regularizers(centerlines, hs):
    # centerlines: [B, K+1, state_dim]
    # hs: [B, K+1, state_dim]
    amp = jnp.mean(jnp.square(hs))

    if centerlines.shape[1] >= 3:
        second_diff = centerlines[:, 2:] - 2.0 * centerlines[:, 1:-1] + centerlines[:, :-2]
        smooth = jnp.mean(jnp.square(second_diff))
    else:
        smooth = 0.0

    return amp, smooth
```

Total dynamics loss에 curved mode일 때만 추가한다.

```math
L_{\mathrm{dyn}}
=
L_{\mathrm{existing}}
+
\lambda_{\mathrm{amp}}L_{\mathrm{center\_amp}}
+
\lambda_{\mathrm{smooth}}L_{\mathrm{center\_smooth}}.
```

---

## 11. Sampling / rollout 수정

기존 rollout:

```python
s_hat_next = existing_mu_i(s_hat_i, s0, sK, ...)
```

curved mode:

```python
c_i = centerline(i; s0, sK, g)
c_next = centerline(i + 1; s0, sK, g)
z_i = s_hat_i - c_i
mu_z = A_i * z_i + kappa_i * (-F_iK * z_i)
mu_star = c_next + mu_z
mu = mu_star + alpha_res * rho_i * residual_net(...)
s_hat_next = mu + optional_noise
```

noise가 기존 구현에 있다면 동일한 방식으로 추가한다.

```python
s_hat_next = mu + tau_noise * sqrt(C_i) * eps
```

hard bridge mode에서는 마지막 transition이 자연스럽게 `sK`에 도달해야 한다. numerical safety가 필요하지 않다면 강제로 overwrite하지 않는다.

---

## 12. Inverse dynamics, critic, actor는 유지

이 기능은 state trajectory proposal을 개선하는 옵션이다. 다음 모듈은 변경하지 않는다.

- inverse dynamics model
- full chunk critic
- partial critic
- scalar value
- Q-Boltzmann proposal
- SPI actor loss

단, bridge rollout output shape이 기존과 동일하도록 보장한다.

---

## 13. Unit tests / smoke tests

가능하면 다음 test를 추가한다.

### Test 1: Backward compatibility

```text
use_curved_centerline=False
```

일 때 기존 bridge output과 동일해야 한다.

### Test 2: Endpoint preservation

```text
use_curved_centerline=True
beta_0=0
beta_K=1
```

일 때

```text
c_0 == s0
c_K == sK
```

이어야 한다.

### Test 3: Zero initialization

```text
centerline_zero_init=True
```

일 때 초기 `h ≈ 0`이고 curved bridge가 기존 straight bridge와 거의 같아야 한다.

### Test 4: Hard final step

hard bridge mode에서 `i=K-1`일 때

```text
mu_star_curved == sK
```

이고, hard variance scaling을 쓰는 exact residual mode에서는

```text
mu_res == sK
```

이어야 한다.

### Test 5: Shape

batch size, state dimension, goal optional case가 모두 정상이어야 한다.

### Test 6: K=1 edge case

NaN이 없어야 하고 smoothness regularizer는 0이거나 skip되어야 한다.

---

## 14. Logging 추가

`use_curved_centerline=True`일 때 다음 값을 log한다.

```text
centerline_amp
centerline_smooth
centerline_deviation
residual_norm
path_loss
rollout_loss
terminal_error
```

정의 예시:

```math
\text{centerline\_deviation}
=
\mathbb{E}_i
\left\|
c_i - ((1-\beta_i)s_0+\beta_i s_K)
\right\|.
```

가능하면 candidate evaluation에서도 다음을 기록한다.

```text
average partial critic score of decoded action chunks
bridge rollout error
inverse dynamics reconstruction error
```

---

## 15. Documentation comment 추가

구현 근처에 다음 설명을 docstring/comment로 추가한다.

```text
Curved centerline bridge:
We retain the analytic linear-SDE bridge, but perform it in residual coordinates around a learned endpoint-preserving centerline c_i. The centerline captures low-frequency path geometry, while the analytic bridge models deviations around it. The beta_i(1-beta_i) envelope guarantees c_0=s0 and c_K=sK. The variance-scaled residual preserves hard endpoint pinning because C_{K-1}=0.
```

---

## 16. Expected behavior

### 옵션 OFF

```text
use_curved_centerline=False
```

기존 DOURI와 동일하게 작동해야 한다.

### 옵션 ON

```text
use_curved_centerline=True
```

bridge mean이 더 이상 단순 straight interpolation에 묶이지 않는다. 모델은 endpoint를 유지하면서 curved state proposal을 학습할 수 있다.

유지되어야 하는 기존 장점:

```text
closed-form analytic one-step teacher
endpoint control
exact-residual mode
inverse-dynamics action decoding
chunk-critic ranking
Q-Boltzmann proposal
SPI actor extraction
```

---

## 17. 최소 변경 원칙

이 기능은 broad rewrite가 아니라 clean optional extension이어야 한다.

구현 우선순위:

1. Config flag 추가.
2. Centerline MLP 추가.
3. Curved bridge teacher 함수 추가.
4. 기존 bridge teacher call site에서 option branch 추가.
5. 기존 residual mode와 결합.
6. path/rollout loss가 새 mean을 쓰도록 연결.
7. centerline regularizer와 logging 추가.
8. smoke tests 추가.

절대 하지 말아야 할 것:

- 기존 bridge behavior를 기본값에서 바꾸기.
- inverse dynamics / critic / actor까지 불필요하게 rewrite하기.
- centerline을 `s_i`에 condition하기.
- finite-gamma beta를 centerline endpoint interpolation에 사용해서 `c_K=sK`를 깨기.

---

## 18. Final mathematical summary

구현할 curved centerline bridge의 최종 수식은 다음이다.

```math
\beta_0=0, \qquad \beta_K=1.
```

```math
c_i
=
(1-\beta_i)s_0
+
\beta_i s_K
+
\beta_i(1-\beta_i)h_\psi(s_0,s_K,g,i/K).
```

```math
z_i=s_i-c_i.
```

```math
\mu_{z,i}^{\star,\gamma}
=
A_i z_i
+
\kappa_i^\gamma(0-F_{i:K}z_i).
```

```math
\mu_{s,i}^{\star,\gamma}
=
c_{i+1}
+
\mu_{z,i}^{\star,\gamma}.
```

```math
\mu_i
=
\mu_{s,i}^{\star,\gamma}
+
\alpha_{\mathrm{res}}\rho_i r_\psi(s_i,s_0,s_K,g,i).
```

where preferably

```math
\rho_i=\sqrt{C_i^\infty}
```

for endpoint safety, or existing

```math
\rho_i=\sqrt{C_i^\gamma}
```

for consistency with current code.

