# Databricks notebook source
# MAGIC %md
# MAGIC # Ray + PyTorch GPU Smoke Test
# MAGIC Minimal notebook — skips SARIMA/GARCH, runs only the Ray+GPU Monte Carlo section.
# MAGIC Used for rapid iteration on GPU debugging without waiting 20+ min for SARIMA/GARCH.

# COMMAND ----------

# ── GPU detection (driver) ────────────────────────────────────────────────────
import subprocess as _subprocess
_gpu_check = _subprocess.run(
    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
    capture_output=True, text=True, timeout=5,
)
HAS_GPU = _gpu_check.returncode == 0 and bool(_gpu_check.stdout.strip())
if HAS_GPU:
    print(f"GPU on driver: {_gpu_check.stdout.strip().splitlines()[0]}")
else:
    print("No GPU on driver (normal for multi-node) — Ray workers will use torch.cuda")

# COMMAND ----------

# ── Ray init on GPU ML cluster ────────────────────────────────────────────────
import ray
from ray.util.spark import setup_ray_cluster, shutdown_ray_cluster

try:
    shutdown_ray_cluster()
except Exception:
    pass

# Prevent Spark from reserving GPUs for its own tasks so Ray can use them all.
spark.conf.set("spark.task.resource.gpu.amount", "0")

setup_ray_cluster(
    max_worker_nodes=1,
    num_cpus_worker_node=2,      # leave 2 of 4 vCPUs free for Spark (g4dn.xlarge)
    num_gpus_worker_node=1,
    collect_log_to_path="/tmp/ray_logs",
)

ray.init(ignore_reinit_error=True)
print(f"Ray initialized | Resources: {ray.cluster_resources()}")

# COMMAND ----------

# ── check_gpu diagnostic ──────────────────────────────────────────────────────
@ray.remote(num_gpus=1)
def _check_gpu_worker():
    import torch, subprocess
    result = {'torch_version': torch.__version__, 'cuda_available': torch.cuda.is_available()}
    print(f"[check_gpu] torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        result['device_count'] = torch.cuda.device_count()
        result['device_name']  = torch.cuda.get_device_name(0)
        print(f"[check_gpu] devices={torch.cuda.device_count()} name={torch.cuda.get_device_name(0)}")
        _a = torch.randn(1000, 1000, device='cuda', dtype=torch.float32)
        _b = (_a @ _a.T).sum().item()
        torch.cuda.synchronize()
        print(f"[check_gpu] matmul smoke-test passed (sum={_b:.2f})")
        result['matmul_ok'] = True
    try:
        nvsmi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        result['nvidia_smi'] = nvsmi.stdout.strip()
        print(f"[check_gpu] nvidia-smi: {nvsmi.stdout.strip()}")
    except Exception as _e:
        result['nvidia_smi_error'] = str(_e)
    return result

print("==> Running GPU diagnostic on a Ray worker...")
_gpu_diag = ray.get(_check_gpu_worker.remote())
print(f"GPU diagnostic result: {_gpu_diag}")
if not _gpu_diag.get('cuda_available', False):
    print("WARNING: Ray workers CANNOT see GPU — Monte Carlo will fall back to CPU!")
else:
    print(f"GPU confirmed on workers: {_gpu_diag.get('device_name', 'unknown')}")

# COMMAND ----------

# ── Monte Carlo task (100% GPU — torch.distributions.StudentT.cdf) ───────────
# Module-level constants and imports: computed/imported once per worker process.
import numpy as np
import torch

_MC_COPULA_DF = 4
_MC_CV        = np.array([0.35, 0.28, 0.42], dtype=np.float32)
_MC_MEANS     = np.array([12.5, 8.3, 5.7],   dtype=np.float32)
_MC_SIGMA2    = np.log(1 + _MC_CV**2)
_MC_MU_LN     = np.log(_MC_MEANS) - _MC_SIGMA2 / 2
_MC_SIGMA_LN  = np.sqrt(_MC_SIGMA2)
_MC_CORR      = np.array([[1.00, 0.40, 0.20],
                           [0.40, 1.00, 0.30],
                           [0.20, 0.30, 1.00]], dtype=np.float32)

# GPU tensor cache: populated once per worker on first GPU call.
_GPU_TENSOR_CACHE: dict = {}

def _get_gpu_tensors(device, dtype):
    key = (str(device), str(dtype))
    if key not in _GPU_TENSOR_CACHE:
        _GPU_TENSOR_CACHE[key] = (
            torch.tensor(_MC_MU_LN,    dtype=dtype, device=device),
            torch.tensor(_MC_SIGMA_LN, dtype=dtype, device=device),
            torch.linalg.cholesky(torch.tensor(_MC_CORR, dtype=dtype, device=device)),
        )
    return _GPU_TENSOR_CACHE[key]

@ray.remote(num_gpus=0.25, num_cpus=0.5)
def simulate_portfolio_losses(n_scenarios: int, seed: int) -> dict:
    import numpy as np
    import torch

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available on this Ray worker")

        device = torch.device('cuda')
        dtype  = torch.float32

        # Retrieve cached tensors (Cholesky computed only on first call per worker)
        mu_t, sig_t, chol_t = _get_gpu_tensors(device, dtype)

        torch.manual_seed(seed)
        z    = torch.randn(n_scenarios, 3,             dtype=dtype, device=device)
        z_nu = torch.randn(n_scenarios, _MC_COPULA_DF, dtype=dtype, device=device)
        chi2 = (z_nu ** 2).sum(dim=1)
        x_cor = z @ chol_t.T
        t_cor = x_cor / (chi2.unsqueeze(1) / _MC_COPULA_DF).sqrt()

        # Step 4: t-CDF via torch.distributions.StudentT (100% CUDA-native).
        _t_dist = torch.distributions.StudentT(
            df=torch.tensor(float(_MC_COPULA_DF), dtype=dtype, device=device)
        )
        u_clip = _t_dist.cdf(t_cor).clamp(1e-6, 1 - 1e-6)

        # Step 5: erfinv + lognormal (all on GPU)
        q      = torch.special.erfinv(2.0 * u_clip - 1.0).mul(2.0 ** 0.5)
        losses = torch.exp(mu_t + sig_t * q)
        total  = losses.sum(dim=1).cpu().numpy().astype(np.float64)
        backend = 'torch-gpu'

    except Exception as e_gpu:
        _gpu_error = f"{type(e_gpu).__name__}: {e_gpu}"
        print(f"[Ray task seed={seed}] GPU path failed: {_gpu_error} — using CPU")
        from scipy.stats import t as tdist, norm as scipy_norm
        chol  = np.linalg.cholesky(_MC_CORR)
        rng   = np.random.default_rng(seed)
        z     = rng.standard_normal((n_scenarios, 3))
        chi2  = rng.chisquare(_MC_COPULA_DF, n_scenarios)
        t_cor = (z @ chol.T) / np.sqrt(chi2[:, None] / _MC_COPULA_DF)
        u     = tdist.cdf(t_cor, df=_MC_COPULA_DF)
        q     = scipy_norm.ppf(np.clip(u, 1e-10, 1 - 1e-10))
        losses = np.exp(_MC_MU_LN + _MC_SIGMA_LN * q)
        total  = losses.sum(axis=1)
        backend = 'numpy-cpu'
    else:
        _gpu_error = None

    return {
        'seed': seed, 'n_scenarios': n_scenarios, 'backend': backend,
        'var_99': float(np.percentile(total, 99)),
        'var_995': float(np.percentile(total, 99.5)),
        'total_loss_mean': float(total.mean()),
        'gpu_error': _gpu_error,
    }

# COMMAND ----------

# ── Launch 4 tasks × 100k scenarios ──────────────────────────────────────────
import time
N_TASKS, N_PER_TASK = 4, 1_000_000   # smoke test — Module 4 production uses 10M/task
print(f"Launching {N_TASKS} Ray tasks x {N_PER_TASK:,} scenarios...")
t0 = time.time()
futures = [simulate_portfolio_losses.remote(N_PER_TASK, seed=42 + i) for i in range(N_TASKS)]
results = ray.get(futures)
elapsed = time.time() - t0

backends = set(r['backend'] for r in results)
print(f"\nCompleted in {elapsed:.1f}s")
print(f"Backends used: {backends}")
print(f"VaR(99.5%): ${sum(r['var_995'] for r in results)/len(results):.1f}M")
print(f"Expected Loss: ${sum(r['total_loss_mean'] for r in results)/len(results):.1f}M")

if backends == {'torch-gpu-hybrid'}:
    print("\nSUCCESS: All tasks used GPU!")
elif 'numpy-cpu' in backends:
    print("\nWARNING: Some or all tasks fell back to CPU — check logs above for error.")

# Export result for get-run-output API capture
_gpu_device = _gpu_diag.get('device_name', 'unknown')
_gpu_errors = [r.get('gpu_error') for r in results if r.get('gpu_error')]
_result_str = (
    f"backends={backends} | "
    f"check_gpu_cuda={_gpu_diag.get('cuda_available')} | "
    f"worker_device={_gpu_device} | "
    f"elapsed={elapsed:.1f}s | "
    f"VaR99.5={sum(r['var_995'] for r in results)/len(results):.1f}M | "
    f"gpu_errors={_gpu_errors} | "
    f"check_gpu_full={_gpu_diag}"
)
print(f"\nEXIT: {_result_str}")
dbutils.notebook.exit(_result_str)
