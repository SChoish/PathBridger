# TRL Appendix 기반 PathBridger 추천 파라미터 정리

이 문서는 **Transitive RL (TRL) appendix / ablation**과 현재 PathBridger의 변형 구조를 함께 고려해, `state-transitive V + local chunk Q + subgoal value bonus` 실험을 위한 권장 loss/config/hyperparameter를 정리한 것이다.

현재 PathBridger 변형의 역할 분리는 다음을 기준으로 한다.

\[
V_\phi(s,g): \text{state-pair transitive value}
\]

\[
Q_\theta(s,A^h,z): \text{SPI actor를 위한 local action-chunk critic}
\]

\[
q_\chi(z\mid s,g): \text{subgoal estimator}
\]

핵심 dependency는 다음과 같아야 한다.

\[
V \rightarrow Q,\qquad V \rightarrow L_{\rm subgoal},\qquad Q \rightarrow L_{\rm SPI}
\]

반대로 `V <- Q`는 이 변형의 main path에서 쓰지 않는다.

---

## 1. Appendix에서 직접 가져온 핵심 근거

### 1.1 TRL value learning recipe

TRL은 triangle inequality 기반으로

\[
V(s,g) \approx \max_z V(s,z)V(z,g)
\]

또는 Q-version으로

\[
Q(s,a,g) \leftarrow Q(s,a,z)Q(z,a',g)
\]

를 사용한다. 실제 구현에서는 다음 세 가지가 중요하다.

1. **in-trajectory / behavioral subgoal만 사용**
   - 임의의 random subgoal은 valid subgoal일 확률이 낮고, 높은 expectile이 필요해져 불안정해진다.
   - 따라서 \(i<k<j\)인 같은 trajectory state \(s_k\)를 split으로 사용한다.

2. **expectile regression으로 max를 approximate**
   - \(\kappa > 0.5\)가 implicit max 역할을 한다.
   - appendix ablation에서는 \(\kappa=0.7\)이 기본적으로 사용된다.

3. **distance-based reweighting**
   - 긴 segment \(s_i\to s_j\) target은 짧은 segment \(s_i\to s_k\), \(s_k\to s_j\) 값에 의존하므로, 짧은 segment를 더 정확히 맞추도록 weight를 둔다.
   - TRL paper의 형태는 대략 estimated distance에 반비례하는 weight이다.

---

## 2. Loss function 추천

## 2.1 State-transitive value \(V(s,g)\)

### Self/base/tri loss 구성

권장 value loss:

\[
L_V
=
\lambda_{\rm self}L_{\rm self}
+
\lambda_{\rm base}L_{\rm base}
+
\lambda_{\rm tri}L_{\rm tri}.
\]

### Self loss

\[
V(s,s)=1.
\]

권장:

```python
loss_v_self = BCEWithLogits(v_self_logits, 1.0)
```

### Base loss

\[
V(s_i,s_{i+d})=\gamma^d,\qquad 1\le d\le H_{\rm base}.
\]

권장:

```python
target_base = gamma ** d
loss_v_base = BCEWithLogits(v_base_logits, target_base)
```

### Transitive loss

\[
y^{\rm tri}_{ij}
=
\operatorname{sg}\left[
\bar V(s_i,s_k)\bar V(s_k,s_j)
\right].
\]

권장:

\[
L_{\rm tri}
=
w_{ij}
D_{\kappa}^{\rm BCE}
\left(
V_\phi(s_i,s_j), y^{\rm tri}_{ij}
\right).
\]

여기서 BCE-expectile은 다음처럼 구현한다.

```python
prob = sigmoid(v_tri_logits)
bce = sigmoid_binary_cross_entropy(v_tri_logits, target_v_tri)
weight = where(target_v_tri >= prob, tau_v, 1.0 - tau_v)
loss_v_tri = weight * bce
```

### Short-leg exact replacement

TRL appendix는 base case를 위해 짧은 leg를 network target 대신 exact discount로 치환한다. 현재 PathBridger에서는 다음을 권장한다.

```python
left_offsets = split_offsets
right_offsets = value_offsets - split_offsets

target_v_left = where(
    left_offsets <= value_base_horizon,
    gamma ** left_offsets,
    target_v_left,
)

target_v_right = where(
    right_offsets <= value_base_horizon,
    gamma ** right_offsets,
    target_v_right,
)
```

strict하게 paper에 맞추려면 `<=1`만 치환해도 되지만, 현 코드에서는 `value_base_horizon=5` 정도까지 치환하는 쪽이 더 안정적일 가능성이 높다.

---

## 2.2 Local action-chunk critic \(Q(s,A^h,z)\)

Local Q는 \(V\)를 만드는 critic이 아니라, SPI actor가 action chunk를 평가하기 위한 critic이다.

권장 target:

\[
y_Q=
\begin{cases}
\gamma^d, & 1\le d\le h,\\
\gamma^h \bar V(s_{i+h},z), & d>h.
\end{cases}
\]

권장 loss:

```python
loss_q_local = BCEWithLogits(q_logits, stop_gradient(target_q))
```

여기에는 expectile을 굳이 쓰지 않는다. \(Q\)는 implicit max용이 아니라 local feasibility / continuation target distillation용이기 때문이다.

---

## 2.3 Subgoal estimator loss

분포형 subgoal estimator:

\[
q_\chi(z\mid s,g)=\mathcal N(\mu_\chi,\Sigma_\chi).
\]

권장 subgoal loss:

\[
L_{\rm sg}
=
-\log q_\chi(z_D\mid s,g)
-
\alpha\,
\mathbb E_{\hat z\sim q_\chi}
[
B(s,\hat z,g)
].
\]

여기서 \(z_D=s_{t+K}\) 또는 clipped target이다.

### 안정성 우선 1단계: transitive product

\[
B_{\rm prod}(s,z,g)=\bar V(s,z)\bar V(z,g).
\]

추천 초기값:

```yaml
subgoal_value_bonus_type: transitive_product
subgoal_value_alpha: 0.1
```

### 성능 확장 2단계: clipped transitive ratio

\[
B_{\rm ratio}(s,z,g)
=
\operatorname{clip}
\left(
\frac{\bar V(s,z)\bar V(z,g)}
{\bar V(s,g)+\epsilon},
0,
c
\right).
\]

추천 초기값:

```yaml
subgoal_value_bonus_type: transitive_ratio
subgoal_value_alpha: 0.03
subgoal_value_ratio_eps: 1.0e-3
subgoal_value_ratio_clip: 5.0
```

`transitive_ratio`는 scale-invariant하다는 장점이 있지만, \(V(s,g)\)가 작을 때 ratio가 폭주할 수 있으므로 반드시 clip을 둔다.

---

## 3. Appendix 기반 기본 hyperparameter

## 3.1 TRL long-horizon OGBench 기준

TRL appendix의 long-horizon OGBench 설정은 다음과 같다.

| 항목 | Appendix 값 | PathBridger 권장 |
|---|---:|---:|
| gradient steps | 1M | `train_epochs` 기준으로 환산 |
| optimizer | Adam | Adam |
| learning rate | 3e-4 | 3e-4 |
| batch size | 1024 | 1024 |
| target update rate | 0.005 | `target_tau: 0.005` |
| discount | 0.999 | 0.995 또는 0.999 |
| MLP size | [1024,1024,1024,1024] | 현재 자원상 [512,512,512] 또는 [512,512,512,512] |
| value goal relabel ratio | `(0,0,1,0)` | TRL mode에서는 same-trajectory future goal 위주 |
| expectile κ | 0.7 | `tau_v: 0.7` |
| distance reweight λ | task별 | 아래 표 참고 |

PathBridger는 현재 `batch_size=1024`, `hidden_dims=(512,512,512)` 계열이 현실적이다. 1B dataset / 대형 네트워크 setting을 그대로 쓰지 않는다면, appendix 숫자는 “상한 reference”로 보고 시작한다.

---

## 3.2 Standard OGBench 기준

TRL appendix의 standard OGBench 설정은 다음과 같다.

| 항목 | Appendix 값 | PathBridger 권장 |
|---|---:|---:|
| gradient steps | 1M | 400 epochs 이상 또는 충분한 update count |
| learning rate | 3e-4 | 3e-4 |
| batch size | 1024 | 1024 |
| MLP size | [512,512,512] | [512,512,512] |
| target update rate | 0.005 | 0.005 |
| discount | 0.99 default, 0.995 humanoidmaze | antmaze/cube/puzzle는 0.995 권장 |
| value goal ratio for TRL/CRL | `(0,1,0,0)` | geometric future goal 또는 same-trajectory future goal |

우리 코드의 TRL sampler는 future same-trajectory goal을 직접 뽑으므로, exact tuple ratio를 완전히 맞추기보다 `value_geom_sample` 여부로 근사하는 것이 현실적이다.

---

## 4. Task-specific \( \kappa,\lambda \) 추천

TRL appendix Table 5 기준, \(\kappa=0.7\)가 거의 모든 task에서 공통으로 사용된다. 따라서 PathBridger도 첫 실험은 다음으로 고정한다.

```yaml
tau_v: 0.7
```

거리 reweight \(\lambda\)는 다음처럼 시작한다.

| 환경군 | appendix λ | PathBridger 추천 |
|---|---:|---:|
| humanoidmaze-giant | 0 | `value_distance_weight_power: 0.0` or `0.5` |
| puzzle-4x5 / 4x6 | 0 | `0.0` or `0.5` |
| pointmaze-large | 0.7 | `0.7` |
| antmaze-large | 0 | `0.0` or `0.5` |
| humanoidmaze-medium | 0 | `0.0` |
| humanoidmaze-large | 0.1 | `0.1` |
| antsoccer-arena | 0.5 | `0.5` |
| cube-single | 0.7 | `0.7` |
| cube-double | 1.0 | `1.0` |
| scene | 1.0 | `1.0` |
| puzzle-3x3 | 0.5 | `0.5` |
| puzzle-4x4 | 2.0 | `1.0` or `2.0` |

PathBridger에서 giant/cube/puzzle를 주로 볼 경우 추천 시작점은 다음이다.

```yaml
# safer
value_distance_weight_power: 0.5

# more TRL-like for cube/puzzle standard
value_distance_weight_power: 1.0
```

현재 구현처럼 `inverse_value_offset`를 기본으로 두고, split-balance term은 ablation으로만 둔다.

```yaml
value_transitive_weight_mode: inverse_value_offset
value_distance_weight_clip_min: 0.05
value_distance_weight_clip_max: 1.0
```

---

## 5. Goal representation 추천

TRL appendix는 oracle representation을 triangle inequality method에 바로 쓰기 어렵다고 보고, 원래 state-goal Q를 학습한 뒤 oracle-conditioned Q로 distillation하는 방식을 사용한다.

따라서 PathBridger의 state-transitive value \(V(s,g)\) 첫 실험은 반드시 다음을 권장한다.

```yaml
critic_agent:
  goal_representation: full

dynamics:
  subgoal_value_goal_representation: full
```

`phi`는 이후 ablation으로 둔다.

```yaml
critic_agent:
  goal_representation: phi
```

이론적으로는 full이 더 깨끗하고, 실용적으로는 cube/puzzle에서 phi가 더 잘 될 가능성도 있으므로 둘 다 비교할 가치가 있다.

---

## 6. Proposal scoring 추천

현재 PathBridger proposal scoring은 다음을 조합한다.

\[
S_{\rm local}=Q(s,A^h,z)
\]

Proposal selection과 SPI actor target 모두 **local Q only**를 사용한다. (`q_plus_v` / `v_only` 옵션은 제거됨.)

---

## 7. Recommended YAML: 안정성 우선

```yaml
critic_agent:
  algorithm: trl
  critic_type: trl
  use_chunk_critic: false

  goal_representation: full
  value_hidden_dims: [512, 512, 512]
  layer_norm: true
  discount: 0.995
  target_tau: 0.005
  q_agg: mean

  tau_v: 0.7
  lambda_v_self: 1.0
  lambda_v_base: 1.0
  lambda_v_tri: 1.0
  lambda_q_local: 1.0

  value_base_horizon: 5
  value_transitive_reweight: true
  value_transitive_weight_mode: inverse_value_offset
  value_distance_weight_power: 0.5
  value_distance_weight_clip_min: 0.05
  value_distance_weight_clip_max: 1.0

  q_value_eps: 1.0e-6
  subgoal_value_ratio_eps: 1.0e-3
  subgoal_value_ratio_clip: 5.0

dynamics:
  subgoal_distribution: diag_gaussian
  subgoal_stochastic_loss: nll
  subgoal_value_bonus_type: transitive_product
  subgoal_value_alpha: 0.1
  subgoal_value_ratio_eps: 1.0e-3
  subgoal_value_ratio_clip: 5.0
  subgoal_value_weight_max: 10.0
  subgoal_value_goal_representation: full
```

---

## 8. Recommended YAML: TRL ratio 본 실험

```yaml
critic_agent:
  algorithm: trl
  critic_type: trl
  use_chunk_critic: false

  goal_representation: full
  discount: 0.995
  target_tau: 0.005
  q_agg: mean

  tau_v: 0.7
  lambda_v_self: 1.0
  lambda_v_base: 1.0
  lambda_v_tri: 1.0
  lambda_q_local: 1.0

  value_base_horizon: 5
  value_transitive_reweight: true
  value_transitive_weight_mode: inverse_value_offset
  value_distance_weight_power: 1.0
  value_distance_weight_clip_min: 0.05
  value_distance_weight_clip_max: 1.0

  subgoal_value_ratio_eps: 1.0e-3
  subgoal_value_ratio_clip: 5.0

dynamics:
  subgoal_distribution: diag_gaussian
  subgoal_stochastic_loss: nll
  subgoal_value_bonus_type: transitive_ratio
  subgoal_value_alpha: 0.03
  subgoal_value_ratio_eps: 1.0e-3
  subgoal_value_ratio_clip: 5.0
  subgoal_value_goal_representation: full
```

---

## 9. Recommended sweep grid

### 9.1 Critic sweep

```yaml
tau_v: [0.5, 0.7, 0.9]
value_distance_weight_power: [0.0, 0.5, 1.0]
goal_representation: [full, phi]
```

우선순위:
1. `tau_v=0.7`
2. `value_distance_weight_power=0.5 or 1.0`
3. `goal_representation=full`

### 9.2 Subgoal bonus sweep

```yaml
subgoal_value_bonus_type: [none, single_value, transitive_product, transitive_ratio]
subgoal_value_alpha:
  none: [0.0]
  single_value: [0.1, 0.3]
  transitive_product: [0.03, 0.1, 0.3]
  transitive_ratio: [0.01, 0.03, 0.1]
```

추천 순서:
1. `none`
2. `single_value`
3. `transitive_product`
4. `transitive_ratio`

---

## 10. 환경별 시작점

### AntMaze / HumanoidMaze 계열

```yaml
discount: 0.995
tau_v: 0.7
value_distance_weight_power: 0.0  # giant는 appendix 기준 0
subgoal_value_bonus_type: transitive_product
subgoal_value_alpha: 0.1
```

### Cube 계열

```yaml
discount: 0.995
tau_v: 0.7
value_distance_weight_power: 0.7  # cube-single 기준
subgoal_value_bonus_type: transitive_product
subgoal_value_alpha: 0.1
```

### Puzzle 계열

Long-horizon puzzle에서는 appendix가 rejection sampling \(N=32\)를 사용한다. PathBridger에서는 `subgoal_num_samples × plan_candidates`를 늘리는 방식으로 근사한다.

```yaml
discount: 0.995
tau_v: 0.7
value_distance_weight_power: 0.5  # 4x5/4x6 appendix는 0, 4x4는 2.0이라 중간값부터
subgoal_num_samples: 4
plan_candidates: 4
subgoal_value_bonus_type: transitive_product
subgoal_value_alpha: 0.1
```

가능하면 다음까지 확장:

```yaml
subgoal_num_samples: 4
plan_candidates: 8
```

---

## 11. 로그 체크리스트

### Value calibration

- `value/self_pred_mean`
  - 빠르게 0.9 이상으로 올라가는지 확인.
- `value/base_pred_mean` vs `value/base_target_mean`
  - 서로 붙어야 함.
- `value/tri_pred_mean` vs `value/tri_target_mean`
  - target이 0으로 붕괴하면 product target이 너무 약함.
- `value/trans_valid_fraction`
  - 너무 낮으면 sampler가 long goal을 충분히 못 뽑는 것.

### Subgoal bonus

- `phase1/subgoal_transitive_product_mean`
- `phase1/subgoal_transitive_ratio_mean`
- `phase1/subgoal_v_s_z_mean`
- `phase1/subgoal_v_z_g_mean`
- `phase1/subgoal_v_s_g_mean`

문제 신호:
- ratio가 항상 clip max에 붙음 → `alpha` 또는 `proposal_v_weight` 감소, `eps` 증가.
- product가 거의 0 → value가 아직 충분히 학습되지 않음. critic warmup 필요.

### Proposal score

- `coupling/proposal_q_score_mean`
- `coupling/proposal_spi_candidate_count`

문제 신호:
- `q_score`가 거의 flat → local Q target이 약하거나 action chunk horizon mismatch.

---

## 12. 구현 patch 권장사항

현재 GitHub 기준으로 추가 권장하는 patch:

1. `_canonicalize_critic_config()`에서 explicit YAML override를 덮어쓰지 않기.
2. `target_v_left/right`의 short leg exact replacement 추가.
3. TRL mode first-run config에서 `goal_representation: full` 명시.
4. `value_transitive_weight_mode` 기본은 `inverse_value_offset`, split-balance는 ablation으로만 사용.

---

## 13. 최종 권장 default

PathBridger TRL first run은 아래를 기본으로 둔다.

```yaml
critic_agent:
  algorithm: trl
  critic_type: trl
  goal_representation: full
  tau_v: 0.7
  lambda_v_self: 1.0
  lambda_v_base: 1.0
  lambda_v_tri: 1.0
  lambda_q_local: 1.0
  value_base_horizon: 5
  value_transitive_reweight: true
  value_transitive_weight_mode: inverse_value_offset
  value_distance_weight_power: 0.5

dynamics:
  subgoal_distribution: diag_gaussian
  subgoal_stochastic_loss: nll
  subgoal_value_bonus_type: transitive_product
  subgoal_value_alpha: 0.1
```

이후 성능이 안정적이면:

```yaml
dynamics:
  subgoal_value_bonus_type: transitive_ratio
  subgoal_value_alpha: 0.03

critic_agent:
  value_distance_weight_power: 1.0
```

로 확장한다.
