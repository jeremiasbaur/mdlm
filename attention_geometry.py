import torch
import torch.nn as nn
import lightning as L

@torch.no_grad()
def compute_symmetry_score(M):
    """
    Computes the symmetry score based on Definition 3.1 and Appendix S1.6[cite: 184, 1074].
    s = 2 * Tr(M @ M) / ||M||_F^2
    """
    # Ensure computation is done in float32 for stability
    M = M.float()
    SYM = 0.5 * (M + M.T)
    M_F_norm_sq = torch.norm(M, p='fro')**2
    
    # Avoid division by zero if the matrix is all zeros
    if M_F_norm_sq < 1e-9:
        return torch.tensor(0.0, device=M.device)
    
    return (torch.norm(SYM, p='fro')**2 / M_F_norm_sq)

@torch.no_grad()
def compute_directionality_score(M, gamma=2.0):
    """
    Computes the directionality score based on Definition 3.2 * -1.
    d = -(r_bar - c_bar) / (r_bar + c_bar)
    
    gamma is a scaling factor for the std dev threshold.
    """
    # Ensure computation is done in float32 for stability
    M = M.float()
    
    col_norms = torch.norm(M, p=2, dim=0)
    row_norms = torch.norm(M, p=2, dim=1)
    
    mu_c, sigma_c = col_norms.mean(), col_norms.std()
    mu_r, sigma_r = row_norms.mean(), row_norms.std()
    
    threshold_c = mu_c + gamma * sigma_c
    threshold_r = mu_r + gamma * sigma_r
    
    # Sum norms of "outlier" rows and columns [cite: 238, 241]
    c_M_bar = torch.sum(col_norms[col_norms > threshold_c])
    r_M_bar = torch.sum(row_norms[row_norms > threshold_r])
    
    denominator = r_M_bar + c_M_bar
    
    # Avoid division by zero
    if denominator < 1e-9:
        score_paper = torch.tensor(0.0, device=M.device)
    else:
        score_paper = ((c_M_bar - r_M_bar) / denominator)

    c_M_bar_excess = torch.sum(col_norms[col_norms > threshold_c] - threshold_c)
    r_M_bar_excess = torch.sum(row_norms[row_norms > threshold_r] - threshold_r)

    if (c_M_bar_excess + r_M_bar_excess) < 1e-9:
        score_excess = torch.tensor(0.0, device=M.device)
    else:
        score_excess = ((c_M_bar_excess - r_M_bar_excess) / (c_M_bar_excess + r_M_bar_excess))

    return score_paper, score_excess

@torch.no_grad()
def compute_spectral_properties(A: torch.Tensor):
    """
    Computes the ratio of the largest to the second-largest absolute eigenvalue
    and the spectral radius of matrix A (max absolute eigenvalue)
    """
    eigvals = torch.linalg.eigvals(A)
    
    # Compute magnitudes and sort descending
    abs_eigvals = torch.sort(torch.abs(eigvals), descending=True).values
    
    lambda_1 = abs_eigvals[0]
    
    if len(abs_eigvals) > 1:
        lambda_2 = abs_eigvals[1]
    else:
        lambda_2 = torch.tensor(1e-10, device=A.device)
    
    if torch.isclose(lambda_2, torch.tensor(0.0, device=A.device)):
        ratio = torch.finfo(A.dtype).max if A.dtype.is_floating_point else torch.finfo(torch.float32).max
        return float(ratio), lambda_1.item()
    else:
        return lambda_1 / lambda_2, lambda_1

@torch.no_grad()
def compute_orthogonality_score(A: torch.Tensor, method: str = "fro"):
    """
    Compute an orthogonality score for matrix A normalized by supplied norm

    Parameters
    ----------
    A : torch.Tensor
        Input matrix.
    method : str, optional
        Method for measuring score:
        - "fro": Frobenius norm of (A^T A - I)
        - "spectral": Spectral norm of (A^T A - I)
        - "sv": Max deviation of singular values from 1
    
    Returns
    -------
    float
        Orthogonality score.
    """
    n = A.shape[1]
    
    I = torch.eye(n, device=A.device, dtype=A.dtype)
    A_n = A / torch.norm(A, p=2, dim=0)

    # Gram matrix calculation
    gram = A_n.T @ A_n
    
    if method == "fro":
        # Frobenius norm
        return torch.linalg.norm(gram - I, ord='fro')
    
    elif method == "spectral":
        # Spectral norm (2-norm)
        return torch.linalg.norm(gram - I, ord=2)
    
    elif method == "sv":
        # Compute only singular values (faster than full SVD)
        s = torch.linalg.svdvals(A)
        return torch.max(torch.abs(s - 1))
    
    else:
        raise ValueError("Unknown method: choose 'fro', 'spectral', or 'sv'")

@torch.no_grad()
def compute_idempotency_error(A: torch.Tensor):
    """
    Computes the Frobenius norm of (A^2 - A). Lower is better (0 means idempotent).
    Quantifies closeness to being a Projection Matrix.
    """
    A_normed = A / torch.norm(A, p='fro')
    A2 = torch.matmul(A_normed, A_normed)
    error_matrix = A2 - A_normed
    return torch.linalg.norm(error_matrix, ord='fro')

@torch.no_grad()
def compute_condition_number(A: torch.Tensor):
    """ 
    Compute the condition number of matrix A
    The condition number measures how much the output value of the function 
    can change for a small change in the input argument (sensitivity).
    """
    # torch.linalg.cond defaults to the 2-norm (spectral norm), same as numpy
    return torch.linalg.cond(A)

@torch.no_grad()
def compute_sparsity_score(A: torch.Tensor, percentile: float = 5.0):
    """
    Compute sparsity of a matrix A (torch tensor) based on a dynamic percentile threshold.
    
    Parameters
    ----------
    A : torch.Tensor
        Input matrix.
    percentile : float, optional
        Percentile (0-100) below which entries are considered zero.
        
    Returns
    -------
    float
        Sparsity score in [0, 1], fraction of 'zeros'.
    """
 
    total = A.numel()
    abs_A = torch.abs(A).flatten()
    
    # 3. Compute the dynamic threshold (q must be between 0 and 1)
    q = percentile / 100.0
    threshold = torch.quantile(abs_A, q)
    
    # 4. Count elements strictly smaller than this threshold
    zeros_count = torch.sum(abs_A < threshold)
    
    return zeros_count / total

@torch.no_grad()
def compute_numerical_sparsity(A: torch.Tensor, tol: float = 1e-6):
    """Checks how many elements are smaller than a fixed tolerance."""
    return (torch.abs(A) < tol).float().mean()

@torch.no_grad()
def compute_trace_norm(A: torch.Tensor):
    """ 
    Compute the trace norm (nuclear norm) of matrix A
    The trace norm is the sum of the singular values ($||A||_* = \sum \sigma_i$).
    """
    # ord='nuc' calculates the nuclear norm directly
    return torch.linalg.norm(A, ord='nuc')

@torch.no_grad()
def compute_frobenius_norm(A: torch.Tensor):
    """ 
    Compute the Frobenius norm of matrix A
    """
    return torch.linalg.norm(A, ord='fro')

@torch.no_grad()
def compute_rank(A: torch.Tensor, tol: float = None):
    """ 
    Compute the numerical rank of matrix A
    """
    if tol is None:
        return torch.linalg.matrix_rank(A).float()
    else:
        return torch.linalg.matrix_rank(A, atol=tol).float()

@torch.no_grad()
def compute_rope_scores(Wq, Wk, n_heads, hidden_dim):
    # Metric Q-K Divergence (Alignment) ---
    # Cosine sim between corresponding rows of Q and K
    q_k_sim = torch.nn.functional.cosine_similarity(Wq, Wk, dim=1)
    s_alignment = q_k_sim.mean()
    
    # Metric RoPE Capacity (Internal Orthogonality) ---
    # We check Row i vs Row i + head_dim/2
    # We need to reshape to [Heads, Head_Dim, Hidden_Dim] to find pairs correctly
    head_dim = hidden_dim // n_heads 
    
    q_per_head = Wq.view(n_heads, head_dim, hidden_dim)
    k_per_head = Wk.view(n_heads, head_dim, hidden_dim)
    
    # We want to compare first half of head to second half
    half_dim = head_dim // 2
    
    # Slice out the halves
    q_half_1 = q_per_head[:, :half_dim, :]
    q_half_2 = q_per_head[:, half_dim:, :]
    
    k_half_1 = k_per_head[:, :half_dim, :]
    k_half_2 = k_per_head[:, half_dim:, :]

    # Calculate similarity between the pairs
    # Result shape: [Heads, Half_Dim]
    pair_sims_q = torch.nn.functional.cosine_similarity(q_half_1, q_half_2, dim=2)
    pair_sims_k = torch.nn.functional.cosine_similarity(k_half_1, k_half_2, dim=2)
    
    # We take the absolute value because -1 is just as bad as 1 for orthogonality
    s_rope_q = pair_sims_q.abs().mean()
    s_rope_k = pair_sims_k.abs().mean()
    # s_rope_q = pair_sims_q.mean()
    # s_rope_k = pair_sims_k.mean()
    return s_rope_q, s_rope_k, s_alignment


class AttentionMatrixMonitor(L.Callback):
    """
    A Lightning Callback to log the symmetry and directionality scores
    of the W_qk = W_q @ W_k.T matrix for each attention block during training.
    """
    def __init__(self, config, log_every_n_steps=1000, gamma=2.0):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.gamma = gamma # Gamma for directionality score threshold
        self.hidden_dim = config.model.hidden_size
        self.n_heads = config.model.n_heads

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Check if it's time to log based on the global step
        if (batch_idx % self.log_every_n_steps) != 0:
            return

        # Access the backbone model (assumed to be pl_module.backbone)
        if not hasattr(pl_module, 'backbone') or not hasattr(pl_module.backbone, 'blocks'):
            Warning("Didn't find backbone of model")
            return

        backbone = pl_module.backbone
        
        all_sym_scores = []
        all_dir_excess_scores = []
        all_dir_paper_scores = []
        all_spectral_radius_scores = []
        all_spectral_gap_scores = []
        all_orthogonality_score = []
        all_idempotency_error = []
        all_condition_number = []
        all_sparsity_score = []
        all_numerical_sparsity = []
        all_trace_norm = []
        all_frobenius_norm = []
        all_rank = []
        all_score_rope_q = [] 
        all_score_rope_k = []
        all_score_alignment  = []

        # Iterate over the model blocks to find attention weights
        for i, block in enumerate(backbone.blocks):

            # This assumes a DiT-like structure with a packed QKV layer
            if not (hasattr(block, 'attn_qkv') and 
                    isinstance(block.attn_qkv, nn.Linear)):
                continue

            # Get the packed QKV weight
            qkv_weight = block.attn_qkv.weight
            
            try:
                # Split into Q, K, V weights (assumes dim 0 is 3 * hidden_dim)
                q_weight, k_weight, _ = torch.chunk(qkv_weight, 3, dim=0)
            except Exception as e:
                pl_module.print(f"Warning: Could not chunk QKV weight for layer {i}. Skipping. Error: {e}")
                continue
            
            # Compute W_qk = W_q @ W_k.T, this is the correct way to calculate
            # it because of the torch nn.Linear layer convention.
            W_qk = q_weight.T @ k_weight
            
            # Compute scores
            sym_score = compute_symmetry_score(W_qk)
            dir_score_paper, dir_score_excess = compute_directionality_score(W_qk, gamma=self.gamma)
            spectral_gap, spectral_radius = compute_spectral_properties(W_qk)
            orthogonality_score = compute_orthogonality_score(W_qk)
            idempotency_error = compute_idempotency_error(W_qk)
            condition_number = compute_condition_number(W_qk)
            sparsity_score = compute_sparsity_score(W_qk)
            numerical_sparsity = compute_numerical_sparsity(W_qk)
            trace_norm = compute_trace_norm(W_qk)
            frobenius_norm = compute_frobenius_norm(W_qk)
            rank = compute_rank(W_qk)
            s_rope_q, s_rope_k, s_alignment = compute_rope_scores(q_weight, k_weight, self.n_heads, self.hidden_dim)
            
            # Log scores for this specific layer
            log_dict = {
                f'layer_{i}/symmetry_score': sym_score,
                f'layer_{i}/directionality_score_paper': dir_score_paper,
                f'layer_{i}/directionality_score_excess': dir_score_excess,
                f'layer_{i}/spectral_gap': spectral_gap,
                f'layer_{i}/spectral_radius': spectral_radius,
                f'layer_{i}/orthogonality_score': orthogonality_score,
                f'layer_{i}/idempotency_error': idempotency_error,
                f'layer_{i}/condition_number': condition_number,
                f'layer_{i}/sparsity_score': sparsity_score,
                f'layer_{i}/numerical_sparsity_score': numerical_sparsity,
                f'layer_{i}/trace_norm': trace_norm,
                f'layer_{i}/frobenius_norm': frobenius_norm,
                f'layer_{i}/rank': rank,
                f'layer_{i}/rope_q_align': s_rope_q,
                f'layer_{i}/rope_k_align': s_rope_k,
                f'layer_{i}/mean_q_k_sim': s_alignment,
            }
            # Log to wandb/etc.
            pl_module.log_dict(log_dict, on_step=True, on_epoch=False, sync_dist=True)
            
            all_sym_scores.append(sym_score)
            all_dir_excess_scores.append(dir_score_excess)
            all_dir_paper_scores.append(dir_score_paper)
            all_spectral_radius_scores.append(spectral_radius)
            all_spectral_gap_scores.append(spectral_gap)
            all_orthogonality_score.append(orthogonality_score)
            all_idempotency_error.append(idempotency_error)
            all_condition_number.append(condition_number)
            all_sparsity_score.append(sparsity_score)
            all_numerical_sparsity.append(numerical_sparsity)
            all_trace_norm.append(trace_norm)
            all_frobenius_norm.append(frobenius_norm)
            all_rank.append(rank)
            all_score_rope_q.append(s_rope_q)
            all_score_rope_k.append(s_rope_k)
            all_score_alignment.append(s_alignment)

        # Also log the median scores across all layers, as seen in the paper's plots [cite: 233, 305]
        stat_func = {'Mean': torch.mean, 'Median': torch.median, 'Min': torch.min, 'Max': torch.max, 'Variance': torch.var}
        if all_sym_scores:
            all_sym = torch.stack(all_sym_scores)
            all_dir_excess = torch.stack(all_dir_excess_scores)
            all_dir_paper = torch.stack(all_dir_paper_scores)
            all_spectral_radius_scores = torch.stack(all_spectral_radius_scores)
            all_spectral_gap_scores = torch.stack(all_spectral_gap_scores)
            all_orthogonality_score = torch.stack(all_orthogonality_score)
            all_idempotency_error = torch.stack(all_idempotency_error)
            all_condition_number = torch.stack(all_condition_number)
            all_sparsity_score = torch.stack(all_sparsity_score)
            all_numerical_sparsity = torch.stack(all_numerical_sparsity)
            all_trace_norm = torch.stack(all_trace_norm)
            all_frobenius_norm = torch.stack(all_frobenius_norm)
            all_rank = torch.stack(all_rank)
            all_score_rope_q = torch.stack(all_score_rope_q)
            all_score_rope_k = torch.stack(all_score_rope_k)
            all_score_alignment = torch.stack(all_score_alignment)

            for name, func in stat_func.items():
                pl_module.log(f'Overall Symmetry/{name}', func(all_sym), on_step=True, on_epoch=False, sync_dist=True)
                pl_module.log(f'Overall Directionality Paper/{name}', func(all_dir_paper), on_step=True, on_epoch=False, sync_dist=True)
                pl_module.log(f'Overall Directionality Excess/{name}', func(all_dir_excess), on_step=True, on_epoch=False, sync_dist=True)
                pl_module.log(f'Overall Spectra Radius/{name}', func(all_spectral_radius_scores), on_step=True, on_epoch=False, sync_dist=True)
                pl_module.log(f'Overall Spectra Gap/{name}', func(all_spectral_gap_scores), on_step=True, on_epoch=False, sync_dist=True)
                pl_module.log(f'Overall Orthogonality/{name}', func(all_orthogonality_score), on_step=True, on_epoch=False, sync_dist=True)
                pl_module.log(f'Overall Idempotency Error/{name}', func(all_idempotency_error), on_step=True, on_epoch=False, sync_dist=True)
                pl_module.log(f'Overall Condition Number/{name}', func(all_condition_number), on_step=True, on_epoch=False, sync_dist=True)
                pl_module.log(f'Overall Sparsity Score/{name}', func(all_sparsity_score), on_step=True, on_epoch=False, sync_dist=True)
                pl_module.log(f'Overall Numerical Sparsity/{name}', func(all_numerical_sparsity), on_step=True, on_epoch=False, sync_dist=True)
                pl_module.log(f'Overall Trace Norm/{name}', func(all_trace_norm), on_step=True, on_epoch=False, sync_dist=True)
                pl_module.log(f'Overall Frobenius Norm/{name}', func(all_frobenius_norm), on_step=True, on_epoch=False, sync_dist=True)
                pl_module.log(f'Overall Rank/{name}', func(all_rank), on_step=True, on_epoch=False, sync_dist=True)
                pl_module.log(f'Overall RoPE W_q Alignment/{name}', func(all_score_rope_q), on_step=True, on_epoch=False, sync_dist=True)
                pl_module.log(f'Overall RoPE W_k Alignment/{name}', func(all_score_rope_k), on_step=True, on_epoch=False, sync_dist=True)
                pl_module.log(f'Overall Mean W_q W_k Similarity/{name}', func(all_score_alignment), on_step=True, on_epoch=False, sync_dist=True)

                