import warnings
import torch
import numpy as np
from torch.nested import nested_tensor

def nested_test(K):
    return K.is_nested

# @torch.compile(fullgraph=True)
def nested_op(x, fn):

    if isinstance(x,list): # Assume two nested tensors
        P, Kb = x

        # Case 1: Both nested
        if nested_test(P) and nested_test(Kb):
            return torch.nested.nested_tensor([
                _safe_apply(fn, p, kb) for p, kb in zip(P.unbind(), Kb.unbind())
            ])

        # Case 2: P regular, Kb nested
        elif not nested_test(P) and nested_test(Kb):
            return torch.nested.nested_tensor([
                _safe_apply(fn, P, kb) for kb in Kb.unbind()
            ])

        # Case 3: P nested, Kb regular
        elif nested_test(P) and not nested_test(Kb):
            return torch.nested.nested_tensor([
                _safe_apply(fn, p, Kb) for p in P.unbind()
            ])

    """Apply operation to nested or regular tensors."""
    if nested_test(x):
        return torch.nested.nested_tensor([_safe_apply(fn, t) for t in x.unbind()])
    return fn(x)

def _safe_apply(fn, *tensors):
    """Apply function with dimension checks"""
    try:
        return fn(tensors)
    except IndexError as e:
        # Handle dimension mismatches
        if any(t.dim() == 1 for t in tensors):
            # Convert 1D to 2D
            tensors = [t.unsqueeze(0) if t.dim() == 1 else t for t in tensors]
            if len(tensors) == 1:
                return fn(tensors[0])
            return fn(tensors)
        raise

# @torch.compile(fullgraph=True)
def nested_mm(A, B):
    """Matrix multiply supporting nested tensors."""
    if nested_test(A):
        return torch.nested.nested_tensor([a.mm(b) for a, b in zip(A.unbind(), B.unbind())])
    return A.mm(B)

# @torch.compile(fullgraph=True)
def nested_sum(x, dim):
    """Sum operation supporting nested tensors."""
    if nested_test(x):
        return torch.stack([t.sum(dim) for t in x.unbind()])
    return x.sum(dim)

# @torch.compile(fullgraph=True)
def nested_max(x):
    """Compute max value of a nested or regular tensor."""
    if nested_test(x):
        return max(t.max() for t in x.unbind())
    return x.max()

def clone(x):
    """Clone tensor or nested tensor."""
    return nested_op(x, lambda t: t.clone())


def nested_max_abs_diff(a, b):
    """Max absolute difference between tensors."""
    if nested_test(a):
        return max(torch.abs(a_t - b_t).max()
               for a_t, b_t in zip(a.unbind(), b.unbind()))
    return torch.abs(a - b).max()


def wimg(p, q, K, epsilon=0.01, maxiter=2000, tol=1e-7, verbose=False,
         f0=0.):
    """Compute the Wasserstein divergence between histograms.

    Parameters
    ----------
    p: numpy array (width, width)
        Must be non-negative.
    q: numpy array (width, width)
        Must be non-negative.
    M: numpy array (width, width)
        One dimensional kernel.
        if None, taken as euclidean gram matrix over [1:n_features]
        normalized by its median.
    epsilon: float > 0.
        Entropy weight. (optional, default 5 / n_features)
    maxiter: int > 0.
        Maximum number of iterations of the Sinkhorn algorithm.
    tol: float >= 0.
        Precision threshold of the Sinkhorn algorithm.
        (optional, default 1e-10)
    returnlog: boolean.
        default False. if True, a list of errors is returned.
    returnmarginal: boolean.
        default False. if True, returns the transport marginal.

    Returns
    -------
    float.
    Wasserstein divergence between p and q.

    """
    bold = torch.ones_like(p, requires_grad=False)
    b = bold.clone()
    Kb = K.mm(K.mm(b).t()).t()
    log = {'cstr': [], 'flag': 0}
    cstr = 10
    for i in range(maxiter):
        a = p / Kb
        Ka = K.t().mm(K.mm(a).t()).t()
        b = q / Ka
        Kb = K.mm(K.mm(b).t()).t()
        with torch.no_grad():
            cstr = abs(Kb * a - p).max().item()
            log["cstr"].append(cstr)

            if cstr < tol:
                break
    if i == maxiter - 1 and verbose:
        warnings.warn("*** Maxiter reached ! err = {} ***".format(cstr))
        log['flag'] = 3
    f = (torch.log(a + 1e-100) * p + torch.log(b + 1e-100) * q).sum() * epsilon
    f += f0
    return f, 0., Kb


def wbarycenter(P, K, epsilon=0.01, maxiter=2000, tol=1e-7, verbose=False,
                f0=0.):
    """Compute the Wasserstein divergence between histograms.

    Parameters
    ----------
    p: numpy array (n_hists, width, width)
        Must be non-negative.
    M: numpy array (width, width)
        One dimensional kernel.
        if None, taken as euclidean gram matrix over [1:n_features]
        normalized by its median.
    epsilon: float > 0.
        Entropy weight. (optional, default 5 / n_features)
    maxiter: int > 0.
        Maximum number of iterations of the Sinkhorn algorithm.
    tol: float >= 0.
        Precision threshold of the Sinkhorn algorithm.
        (optional, default 1e-10)
    returnlog: boolean.
        default False. if True, a list of errors is returned.
    returnmarginal: boolean.
        default False. if True, returns the transport marginal.

    Returns
    -------
    float.
    Wasserstein barycenter of P.

    """
    n_hists, width, _ = P.shape
    bold = torch.ones_like(P, requires_grad=False)
    b = bold.clone()
    Kb = convol_imgs(b, K)
    log = {'cstr': [], 'obj': [], 'flag': 0}
    cstr = 10
    for i in range(maxiter):
        a = P / Kb
        Ka = convol_imgs(a, K.t())
        q = np.prod(Ka, dim=0) ** (1 / n_hists)
        Q = q[None, :, :]
        b = Q / Ka
        Kb = convol_imgs(b, K)
        with torch.no_grad():
            cstr = abs(a * Kb - P).max().item()
            log["cstr"].append(cstr)

            if cstr < tol and 0:
                break
    if i == maxiter - 1 and verbose:
        warnings.warn("*** Maxiter reached ! err = {} ***".format(cstr))
        log['flag'] = 3
    f = (torch.log(a + 1e-100) * P + torch.log(b + 1e-100) * Q).sum()
    f *= epsilon
    f += f0
    return q, f


def wbarycenterkl(P, K, epsilon=0.01, gamma=1, maxiter=2000, tol=1e-7,
                  verbose=False, f0=0.):
    """Compute the Wasserstein divergence between histograms.

    Parameters
    ----------
    p: numpy array (n_hists, width, width)
        Must be non-negative.
    M: numpy array (width, width)
        One dimensional kernel.
        if None, taken as euclidean gram matrix over [1:n_features]
        normalized by its median.
    epsilon: float > 0.
        Entropy weight. (optional, default 5 / n_features)
    maxiter: int > 0.
        Maximum number of iterations of the Sinkhorn algorithm.
    tol: float >= 0.
        Precision threshold of the Sinkhorn algorithm.
        (optional, default 1e-10)
    returnlog: boolean.
        default False. if True, a list of errors is returned.
    returnmarginal: boolean.
        default False. if True, returns the transport marginal.

    Returns
    -------
    float.
    Wasserstein barycenter of P.

    """
    n_hists, width, _ = P.shape
    bold = torch.ones_like(P, requires_grad=False)
    b = bold.clone()
    Kb = convol_imgs(b, K)
    log = {'cstr': [], 'obj': [], 'flag': 0}
    cstr = 10
    fi = gamma / (gamma + epsilon)
    for i in range(maxiter):
        a = (P / Kb) ** fi
        Ka = convol_imgs(a, K.t())
        q = ((Ka ** (1 - fi)).mean(dim=0))
        q = q ** (1 / (1 - fi))
        Q = q[None, :, :]
        b = (Q / Ka) ** fi
        Kb = convol_imgs(b, K)
        with torch.no_grad():
            cstr = abs(a * Kb - P).max().item()
            log["cstr"].append(cstr)

            if cstr < tol:
                break
    if i == maxiter - 1 and verbose:
        warnings.warn("*** Maxiter reached ! err = {} ***".format(cstr))
        log['flag'] = 3
    plsum = (a * Kb).sum()
    f = - (epsilon + 2 * gamma) * plsum + gamma * (P.sum())
    f += f0
    return q, f


def convol_imgs(imgs, K):
    if nested_test(K):
        # K is a nested tensor
        def nested_einsum(imgs_nt, K_nt):
            """
            Perform einsum operations on nested tensors with compatible shapes.
            K_nt: Nested tensor of kernels (each ...ij may have different sizes)
            imgs_nt: Nested tensor of images (each ...jl may have different sizes)
            """
            # Ensure we have pairs of sub-tensors
            assert len(K_nt) == len(imgs_nt)

            results = []
            for k, img in zip(K_nt.unbind(), imgs_nt.unbind()):
                # First einsum: ...ij,kjl->kil
                kx = torch.einsum("ij,jl->il", k, img)  # Adjusted for 2D case

                # Second einsum: ...ij,klj->kli
                kxy = torch.einsum("ij,lj->li", k, kx)  # Adjusted for 2D case

                results.append(kxy)

            return nested_tensor(results)

        compiled_einsum = torch.compile(nested_einsum, fullgraph=True)
        kxy = compiled_einsum(imgs, K)
    else:
        kx = torch.einsum("...ij,kjl->kil", K, imgs)
        kxy = torch.einsum("...ij,klj->kli", K, kx)
    return kxy


#? Deceprecated function, kept for backward compatibility
def convol_old(imgs, K):
    kxy = torch.zeros_like(imgs)
    for i, img in enumerate(imgs):
        kxy[i] = K.mm(K.mm(img).t()).t()
    return kxy


def wimg_parallel(p, Q, K, epsilon=0.01, maxiter=2000, tol=1e-7,
                  verbose=False, f0=0.):
    """Compute the Wasserstein divergence between histograms.

    Parameters
    ----------
    p: numpy array (width, width)
        Must be non-negative.
    Q: numpy array (n_imgs, width, width)
        Must be non-negative.
    M: numpy array (width, width)
        One dimensional kernel.
        if None, taken as euclidean gram matrix over [1:n_features]
        normalized by its median.
    epsilon: float > 0.
        Entropy weight. (optional, default 5 / n_features)
    maxiter: int > 0.
        Maximum number of iterations of the Sinkhorn algorithm.
    tol: float >= 0.
        Precision threshold of the Sinkhorn algorithm.
        (optional, default 1e-10)
    returnlog: boolean.
        default False. if True, a list of errors is returned.
    returnmarginal: boolean.
        default False. if True, returns the transport marginal.

    Returns
    -------
    float.
    Wasserstein divergence between p and q.

    """
    if Q.ndimension() == 2:
        Q = Q[None, :, :]
    bold = torch.ones_like(Q, requires_grad=False)
    b = bold.clone()
    Kb = convol_imgs(b, K)
    log = {'cstr': [], 'obj': [], 'flag': 0, 'a': [], 'b': []}
    cstr = 10
    for i in range(maxiter):
        a = p / Kb
        Ka = convol_imgs(a, K.t())
        b = Q / Ka
        Kb = convol_imgs(b, K)
        with torch.no_grad():
            cstr = abs(Kb * a - p).mean().item()
            log["cstr"].append(cstr)

            if cstr < tol:
                break
    if i == maxiter - 1 and verbose:
        warnings.warn("*** Maxiter reached ! err = {} ***".format(cstr))
        log['flag'] = 3
    f = (torch.log(a + 1e-100) * p + torch.log(b + 1e-100) * Q).sum(dim=(1, 2))
    f *= epsilon
    f += f0

    return f, a, Kb


def wkl(p, Q, K, epsilon=0.01, gamma=1., maxiter=2000, tol=1e-7,
        verbose=False, f0=0., compute_grad=False, Kb=None):
    """Compute the Wasserstein divergence between histograms.

    Parameters
    ----------
    p: numpy array (width, width)
        Must be non-negative.
    Q: numpy array (n_imgs, width, width)
        Must be non-negative.
    M: numpy array (width, width)
        One dimensional kernel.
        if None, taken as euclidean gram matrix over [1:n_features]
        normalized by its median.
    epsilon: float > 0.
        Entropy weight. (optional, default 5 / n_features)
    maxiter: int > 0.
        Maximum number of iterations of the Sinkhorn algorithm.
    tol: float >= 0.
        Precision threshold of the Sinkhorn algorithm.
        (optional, default 1e-10)
    returnlog: boolean.
        default False. if True, a list of errors is returned.
    returnmarginal: boolean.
        default False. if True, returns the transport marginal.

    Returns
    -------
    float.
    Wasserstein divergence between p and q.

    """
    Q = Q.t()
    if Q.ndimension() == 1:
        Q = Q[:, None]
    Qs = Q.sum(dim=0)
    psum = p.sum()
    f = gamma * (Qs + psum)
    idx = Qs > -1
    if compute_grad:
        g = gamma * (1 - 10 ** (20 * epsilon / gamma)) * torch.ones_like(Q)
    Q = Q[:, idx]
    if gamma == 0.:
        return wimg_parallel(p, Q, K, epsilon, maxiter, tol, verbose)
    if psum < -1:
        return f, g
    fi = gamma / (gamma + epsilon)

    log = {'cstr': [], 'obj': [], 'flag': 0, 'a': [], 'b': []}
    cstr = 10
    bold = torch.ones_like(Q, requires_grad=False)
    if Kb is None:
        Kb = K.mm(bold)
    for i in range(maxiter):
        a = (p[:, None] / Kb) ** fi
        Ka = K.t().mm(a)
        b = (Q / Ka) ** fi
        Kb = K.mm(b)
        with torch.no_grad():
            if i % 10 == 0:
                cstr = abs(bold - b).max()
                cstr /= max(1, abs(bold).max(), abs(b).max())
            bold = b.clone()
            log["cstr"].append(cstr)

            if cstr < tol:
                break
    if i == maxiter - 1 and verbose:
        warnings.warn("*** Maxiter reached ! err = {} ***".format(cstr))
        log['flag'] = 3
    plsum = (a * Kb).sum(0)
    f[idx] += epsilon * (f0 - plsum)
    f[idx] += - 2 * gamma * plsum
    # M = - epsilon * torch.log(K)
    # f = (convol_imgs(b, M * K) * a).sum((1, 2))
    if compute_grad:
        g = gamma * (1 - a ** (- epsilon / gamma))
        return f, g.t(), Kb
    return f, 0., Kb


def wkllog(p, q, K, epsilon=0.01, gamma=1., maxiter=2000, tol=1e-7,
           verbose=False, f0=0., compute_grad=False):
    """Compute the Wasserstein divergence between histograms.

    Parameters
    ----------
    p: numpy array (width, width)
        Must be non-negative.
    Q: numpy array (n_imgs, width, width)
        Must be non-negative.
    M: numpy array (width, width)
        One dimensional kernel.
        if None, taken as euclidean gram matrix over [1:n_features]
        normalized by its median.
    epsilon: float > 0.
        Entropy weight. (optional, default 5 / n_features)
    maxiter: int > 0.
        Maximum number of iterations of the Sinkhorn algorithm.
    tol: float >= 0.
        Precision threshold of the Sinkhorn algorithm.
        (optional, default 1e-10)
    returnlog: boolean.
        default False. if True, a list of errors is returned.
    returnmarginal: boolean.
        default False. if True, returns the transport marginal.

    Returns
    -------
    float.
    Wasserstein divergence between p and q.

    """
    if gamma == 0.:
        return wimg_parallel(p, q, K, epsilon, maxiter, tol, verbose)
    fi = gamma / (gamma + epsilon)

    log = {'cstr': [], 'obj': [], 'flag': 0, 'a': [], 'b': []}
    cstr = 10
    if compute_grad:
        g = (1 - 10 ** (20 * epsilon / gamma)) * torch.ones_like(p)
    ps = p.sum()
    qs = q.sum()
    if ps < -1 or qs < -1:
        return gamma * (qs + ps), g
    # if support:
    support_p = p > -1e-10
    support_q = q > -1e-10

    p = torch.log(p + 1e-15)
    q = torch.log(q + 1e-15)
    K = K - 1e-100
    p = p[support_p]
    q = q[support_q]

    K = K[support_p]
    K = K[:, support_q]
    b = torch.zeros_like(q, requires_grad=False)
    Kb = torch.logsumexp(K + b[None, :], dim=1)
    psumold = 0.
    for i in range(maxiter):
        a = fi * (p - Kb)
        Ka = torch.logsumexp(K.t() + a[None, :], dim=1)
        b = fi * (q - Ka)
        Kb = torch.logsumexp(K + b[None, :], dim=1)
        psum = torch.exp(a + Kb).sum().item()
        with torch.no_grad():
            cstr = abs(psumold - psum) / max(1, psumold, psum)
            psumold = psum
            log["cstr"].append(cstr)

            if cstr < tol:
                break
    if i == maxiter - 1 and verbose:
        warnings.warn("*** Maxiter reached ! err = {} ***".format(cstr))
        log['flag'] = 3
    f = epsilon * (f0 - psum)
    f = f + gamma * (ps + qs - 2 * psum)
    # M = - epsilon * torch.log(K)
    # f = (convol_imgs(b, M * K) * a).sum((1, 2))
    if compute_grad:
        g = 1 - torch.exp(- epsilon * a / gamma)
        g *= gamma
        return f, g, Kb
    return f, 0, Kb


def negentropy_img(P, K, epsilon=0.01, gamma=1., maxiter=100, tol=1e-7,
                   verbose=False, f0=0., compute_grad=False, a=None):
    """Compute the Negentropy term W(p, p) elementwise.

    Parameters
    ----------
    P: numpy array (n_hists, width, width)
        Must be non-negative.
    M: numpy array (width, width)
        One dimensional kernel.
        if None, taken as euclidean gram matrix over [1:n_features]
        normalized by its median.
    epsilon: float > 0.
        Entropy weight. (optional, default 5 / n_features)
    maxiter: int > 0.
        Maximum number of iterations of the Sinkhorn algorithm.
    tol: float >= 0.
        Precision threshold of the Sinkhorn algorithm.
        (optional, default 1e-10)
    returnlog: boolean.
        default False. if True, a list of errors is returned.
    returnmarginal: boolean.
        default False. if True, returns the transport marginal.

    Returns
    -------
    float.
    Wasserstein divergence between p and q.

    """
    if P.ndimension() == 2:
        P = P[None, :, :]
    if gamma == 0.:
        fi = 1.
    else:
        fi = gamma / (gamma + epsilon)
    if a is None:
        a = torch.ones_like(P, requires_grad=False)
    aold = a.clone()
    log = {'cstr': [], 'obj': [], 'flag': 0}
    cstr = 10
    # torch.set_grad_enabled(grad)
    for i in range(maxiter):
        Ka = convol_imgs(a, K)
        a = a ** 0.5 * (P / Ka) ** (fi / 2)

        with torch.no_grad():
            cstr = abs(a - aold).max() / max(1, a.max(), aold.max())
            aold = a.clone()
            log["cstr"].append(cstr)

            if cstr < tol:
                break
    if i == maxiter - 1 and verbose:
        warnings.warn("*** Maxiter reached ! err = {} ***".format(cstr))
        log['flag'] = 3
    if gamma:
        psum = (a * Ka).sum((1, 2))
        f = - epsilon * psum
        f = f + gamma * (2 * P.sum((1, 2)) - 2 * psum)
    else:
        f = 2 * (P * torch.log(a + 1e-100)).sum((1, 2))
    f = f + epsilon * f0
    if compute_grad:
        grad = gamma * (1 - a ** (- epsilon / gamma))
        return f, grad, a
    return f, 0., a


def negentropy(P, K, epsilon=0.01, gamma=1., maxiter=100, tol=1e-7,
               verbose=False, f0=0., compute_grad=False, a=None):
    """Compute the Negentropy term W(p, p) elementwise.

    Parameters
    ----------
    P: numpy array (n_hists, width, width)
        Must be non-negative.
    M: numpy array (width, width)
        One dimensional kernel.
        if None, taken as euclidean gram matrix over [1:n_features]
        normalized by its median.
    epsilon: float > 0.
        Entropy weight. (optional, default 5 / n_features)
    maxiter: int > 0.
        Maximum number of iterations of the Sinkhorn algorithm.
    tol: float >= 0.
        Precision threshold of the Sinkhorn algorithm.
        (optional, default 1e-10)
    returnlog: boolean.
        default False. if True, a list of errors is returned.
    returnmarginal: boolean.
        default False. if True, returns the transport marginal.

    Returns
    -------
    float.
    Wasserstein divergence between p and q.

    """

    nested_mode_P =nested_test(P)
    nested_mode_K = nested_test(K)

    P = nested_op(P, lambda x:x.t())

    if P.ndimension() == 1:
        P = P[:, None]
    if gamma == 0.:
        fi = 1.
    else:
        fi = gamma / (gamma + epsilon)
    aold = nested_op(P,torch.ones_like)

    a = clone(a)

    log = {'cstr': [], 'obj': [], 'flag': 0}
    cstr = 10
    # torch.set_grad_enabled(grad)
    for i in range(maxiter):
        Ka = nested_mm(K,a)
        a = nested_op([a, P, Ka],
                             lambda x: x[0]**0.5 * (x[1]/x[2])**(fi/2))

        with torch.no_grad():
            cstr = nested_max_abs_diff(a,aold) / max(1, nested_max(a), nested_max(aold))
            aold = clone(a)
            log["cstr"].append(cstr)

            if cstr < tol:
                break
    if i == maxiter - 1 and verbose:
        warnings.warn("*** Maxiter reached ! err = {} ***".format(cstr))
        log['flag'] = 3
    if gamma:
        psum = nested_sum(a*Ka,0) #** Not sure this will work
        f = - epsilon * psum
        f = f + gamma * (2 * nested_sum(P,0) - 2 * psum)
    else:
        f = 2 * nested_sum(
            nested_op([P, a],
            lambda x: x[0] * torch.log(x[1] + 1e-100)),  # Numerical stability
            0
        )
    f = f + epsilon * f0
    if compute_grad:
        grad = nested_op(a,lambda x: gamma * (1 - x ** (- epsilon / gamma)))
        return f, nested_op(grad,lambda x: x.t()), a
    return f, 0., a

def negentropy_log_(p, K, epsilon=0.01, gamma=1., maxiter=2000, tol=1e-7,
                    verbose=False, f0=0., compute_grad=False, a=None):
    """Compute the Negentropy term W(p, p) elementwise.

    Parameters
    ----------
    P: numpy array (n_hists, width, width)
        Must be non-negative.
    M: numpy array (width, width)
        One dimensional kernel.
        if None, taken as euclidean gram matrix over [1:n_features]
        normalized by its median.
    epsilon: float > 0.
        Entropy weight. (optional, default 5 / n_features)
    maxiter: int > 0.
        Maximum number of iterations of the Sinkhorn algorithm.
    tol: float >= 0.
        Precision threshold of the Sinkhorn algorithm.
        (optional, default 1e-10)
    returnlog: boolean.
        default False. if True, a list of errors is returned.
    returnmarginal: boolean.
        default False. if True, returns the transport marginal.

    Returns
    -------
    float.
    Wasserstein divergence between p and q.

    """

    if gamma == 0.:
        fi = 1.
    else:
        fi = gamma / (gamma + epsilon)
    if compute_grad:
        grad = (1 - 10 ** (20 * epsilon / gamma)) * torch.ones_like(p)
    ps = p.sum()
    # if support:
    support_p = p > -1e-10
    p = p[support_p]
    logp = torch.log(p + 1e-10)
    K = K - 1e-100
    grad = (1 - 10 ** (20 * epsilon / gamma)) * torch.ones_like(p)

    K = K[support_p][:, support_p]
    aold = torch.zeros_like(p, requires_grad=False)
    a = aold.clone()
    log = {'cstr': [], 'obj': [], 'flag': 0}
    cstr = 10
    psumold = 0.
    for i in range(maxiter):
        ka = torch.logsumexp(K + a[None, :], dim=1)
        a = 0.5 * (a + fi * (logp - ka))
        psum = torch.exp(a + ka).sum().item()

        with torch.no_grad():
            cstr = abs(psumold - psum) / max(1, psumold, psum)
            psumold = psum
            log["cstr"].append(cstr)

            if cstr < tol:
                break
    if i == maxiter - 1 and verbose:
        warnings.warn("*** Maxiter reached ! err = {} ***".format(cstr))
        log['flag'] = 3
    if gamma:
        f = - epsilon * psum
        f += gamma * (2 * ps - 2 * psum)
    else:
        f = 2 * (p * a).sum()
    f += epsilon * f0
    if compute_grad:
        grad[support_p] = 1 - torch.exp(- epsilon * a / gamma)
        grad *= gamma
        return f, grad, a
    return f, 0., a


def negentropy_log(P, K, epsilon=0.01, gamma=1., maxiter=100, tol=1e-7,
                   verbose=False, f0=0., compute_grad=False, a=None):
    if P.ndimension() == 1:
        P = P[None, :]
    n_times, dimension = P.shape
    f = torch.zeros(n_times, dtype=P.dtype, device=P.device)
    grad = torch.zeros_like(P)
    for i, p in enumerate(P):
        ff, gg = negentropy_log_(p, K, epsilon, gamma, maxiter, tol,
                                 verbose, f0, compute_grad=compute_grad)
        f[i] = ff
        grad[i] = gg
    if compute_grad:
        return f, grad, a
    return f, 0., a


def wimgkl(p, Q, K, epsilon=0.01, gamma=1., maxiter=2000, tol=1e-7,
           verbose=False, f0=0., compute_grad=False, Kb=None):
    """Compute the Wasserstein divergence between histograms.

    Parameters
    ----------
    p: numpy array (width, width)
        Must be non-negative.
    Q: numpy array (n_imgs, width, width)
        Must be non-negative.
    M: numpy array (width, width)
        One dimensional kernel.
        if None, taken as euclidean gram matrix over [1:n_features]
        normalized by its median.
    epsilon: float > 0.
        Entropy weight. (optional, default 5 / n_features)
    maxiter: int > 0.
        Maximum number of iterations of the Sinkhorn algorithm.
    tol: float >= 0.
        Precision threshold of the Sinkhorn algorithm.
        (optional, default 1e-10)
    returnlog: boolean.
        default False. if True, a list of errors is returned.
    returnmarginal: boolean.
        default False. if True, returns the transport marginal.

    Returns
    -------
    float.
    Wasserstein divergence between p and q.

    """
    if Q.ndimension() == 2:
        Q = Q[None, :, :]
    Qs = Q.sum(dim=(1, 2))
    psum = p.sum()
    f = gamma * (Qs + psum)

    if gamma == 0.:
        return wimg_parallel(p, Q, K, epsilon, maxiter, tol, verbose)
    fi = gamma / (gamma + epsilon)

    log = {'cstr': [], 'obj': [], 'flag': 0, 'a': [], 'b': []}
    cstr = 10
    b = torch.ones_like(Q, requires_grad=False)
    if Kb is None:
        Kb = convol_imgs(b, K)
    bold = b.clone()
    for i in range(maxiter):
        a = (p / Kb) ** fi
        Ka = convol_imgs(a, K.t())
        b = (Q / Ka) ** fi
        Kb = convol_imgs(b, K)
        with torch.no_grad():
            if i % 10 == 0:
                cstr = abs(bold - b).max()
                cstr /= max(1, abs(bold).max(), abs(b).max())
            bold = b.clone()
            log["cstr"].append(cstr)

            if cstr < tol:
                break

            if cstr < tol or torch.isnan(psum).any():
                break
    if i == maxiter - 1:
        warnings.warn("*** Maxiter reached ! err = {} ***".format(cstr))
        log['flag'] = 3
    plsum = (a * Kb).sum((1, 2))

    f += epsilon * (f0 - plsum)
    f += - 2 * gamma * plsum
    if compute_grad:
        g = gamma * (1 - a ** (- epsilon / gamma))
        return f, g, Kb
    return f, 0., Kb


def convol_huge_imgs(imgs, K):
    n, m, dimension, dimension = imgs.shape
    out = convol_imgs(imgs.reshape(n * m, dimension, dimension), K)
    out = out.reshape(n, m, dimension, dimension)
    return out

def convol_huge(imgs, K):
    if nested_test(K):
        print("test")
        out = []
        for t in range(len(imgs.unbind())):
            k = K[t].t()
            img_t = imgs[t]
            print(k,img_t)
            # print(img_t.shape)
            # img_flat = img_t.flatten().unsqueeze(0)
            out.append(k @ img_t)
        return nested_tensor(out)
    else:
        time_dimension, n, m = imgs.shape #? IMG has fixed spatial dimensions
        imgs_reshape = imgs.reshape(time_dimension, n * m)
        out = K @ imgs_reshape
        out = out.reshape(time_dimension, n, m)
        return out

def convol_huge_log(imgs, C):
    if nested_test(C):
        out = []
        for t in range(len(imgs.unbind())):
            c = C[t].unsqueeze(0)
            img_flat = imgs[t].flatten().unsqueeze(0)
            out_t=torch.logsumexp(c.unsqueeze(-1) + img_flat.unsqueze(1), dim=1)
            out.append(out_t @ img_flat)
        return nested_tensor(out)
    else:
        dimension, n, m = imgs.shape
        imgs = imgs.reshape(dimension, n * m)
        out = torch.logsumexp(C[:, :, None] + imgs[None, :], dim=1)
        out = out.reshape(dimension, n, m)
    return out

def convol_imgs_log(imgs, C):
    """Compute log separable kernel application with nested tensor support."""
    if nested_test(C):
        out = []
        for i in range(len(imgs.unbind())):  # Loop over batch dimension
            img = imgs[i]  # Shape: (dimension, dimension)
            c = C[i].unsqueeze(0)  # Shape: (1, c_dim)

            # First logsumexp
            x = torch.logsumexp(c[None, :, :, None] + img[:, None, None, :], dim=-1)
            # Second logsumexp
            x = torch.logsumexp(c.transpose(0, 1)[:, :, None] + x[:, :, None], dim=1)
            out.append(x)
        return nested_tensor(out)
    else:
        n, dimension, _ = imgs.shape
        # First logsumexp
        x = torch.logsumexp(C[None, None, :, :] + imgs[:, :, None], dim=-1)
        # Second logsumexp
        x = torch.logsumexp(C.t()[None, :, :, None] + x[:, :, None], dim=1)
        return x.reshape(n, dimension, dimension)

def convol_huge_imgs_log(imgs, C):
    """Compute log separable kernel application for huge images with nested support."""
    if nested_test(C):
        out = []
        for i in range(len(imgs.unbind())):  # Loop over first batch dimension
            for j in range(len(imgs[i])):  # Loop over second batch dimension
                img = imgs[i][j]  # Shape: (dimension, dimension)
                c = C[i].unsqueeze(0)  # Shape: (1, c_dim)

                # First logsumexp
                x = torch.logsumexp(c[None, :, :, None] + img[:, None, None, :], dim=-1)
                # Second logsumexp
                x = torch.logsumexp(c.transpose(0, 1)[:, :, None] + x[:, :, None], dim=1)
                out.append(x)
        return nested_tensor(out).reshape(len(imgs.unbind()), len(imgs[0]), -1, -1)
    else:
        n, m, dimension, _ = imgs.shape
        imgs = imgs.reshape(n * m, dimension, dimension)
        # First logsumexp
        x = torch.logsumexp(C[None, None, :, :] + imgs[:, :, None], dim=-1)
        # Second logsumexp
        x = torch.logsumexp(C.t()[None, :, :, None] + x[:, :, None], dim=1)
        return x.reshape(n, m, dimension, dimension)


def monster_img(P, Q, K, epsilon=0.01, gamma=1., maxiter=2000, tol=1e-8,
                verbose=False, f0=0., compute_grad=False, Kb=None):
    """
    Compute the convolutional Sinkhorn-based Wasserstein divergence
    between 2D probability distributions (e.g., images), using a
    shared cost kernel `K`.

    Parameters
    ----------
    P : torch.Tensor, shape (n_P, H, W) or (H, W)
        First batch or single input of 2D histograms (non-negative).
    Q : torch.Tensor, shape (n_Q, H, W) or (H, W)
        Second batch or single input of 2D histograms (non-negative).
    K : torch.Tensor, shape (H, W)
        Convolution kernel (e.g., Gaussian based on cost matrix).
    epsilon : float, default=0.01
        Entropic regularization parameter.
    gamma : float, default=1.0
        KL-divergence weight for marginal relaxation.
    maxiter : int, default=2000
        Maximum number of Sinkhorn iterations.
    tol : float, default=1e-8
        Convergence tolerance.
    verbose : bool, default=False
        If True, display warning if max iterations are reached.
    f0 : float, default=0.0
        Optional additive constant for initial cost matrix.
    compute_grad : bool, default=False
        If True, return gradients with respect to P.
    Kb : torch.Tensor, optional
        Precomputed right-marginal kernel convolution.

    Returns
    -------
    f : torch.Tensor, shape (n_P, n_Q)
        Wasserstein divergence matrix between each P[i] and Q[j].
    g : torch.Tensor or float
        Gradient of f w.r.t. P (if compute_grad=True), else 0.
    Kb : torch.Tensor
        Final right-marginal kernel convolution (can be reused).
    """

    if Q.ndimension() == 2:
        Q = Q[None, :, :]
    if P.ndimension() == 2:
        P = P[None, :, :]
    Qs = Q.sum(dim=(1, 2))
    Ps = P.sum(dim=(1, 2))

    f = gamma * (Qs[None, :] + Ps[:, None]) + epsilon * f0
    idq = Qs > -1e-2
    idp = Ps > -1e-2
    if compute_grad:
        g = torch.ones(len(P), *Q.shape, dtype=Q.dtype, device=Q.device)
        g *= gamma * (1 - 10 ** (20 * epsilon / gamma))
    Q = Q[idq]
    P = P[idp]

    fi = gamma / (gamma + epsilon)

    log = {'cstr': [], 'obj': [], 'flag': 0, 'a': [], 'b': []}
    cstr = 10
    if Kb is None:
        b = torch.ones_like(Q)[None, :]
        Kb = convol_huge_imgs(b, K)

    bold = b.clone()
    for i in range(maxiter):
        a = (P[:, None, :, :] / Kb) ** fi
        Ka = convol_huge_imgs(a, K.t())
        b = (Q[None, :, :, :] / Ka) ** fi
        Kb = convol_huge_imgs(b, K)

        with torch.no_grad():
            if i % 10 == 0:
                cstr = abs(bold - b).max()
                cstr /= max(1, abs(bold).max(), abs(b).max())
            bold = b.clone()
            log["cstr"].append(cstr)

            if cstr < tol:
                break
            if torch.isnan(Kb).any():
                warnings.warn("Numerical Errors ! Stopped at last stable "
                              "iteration.")
                break
    if i == maxiter - 1 and verbose:
        warnings.warn("*** Maxiter reached ! err = {} ***".format(cstr))
        log['flag'] = 3
    plsum = (a * Kb).sum((2, 3))

    for j in range(len(Q)):
        if idq[j]:
            f[idp, j] += - (epsilon + 2 * gamma) * plsum[:, j]

    if compute_grad:
        for j in range(len(Q)):
            if idq[j]:
                g[idp, j] = gamma * (1 - a[:, j] ** (- epsilon / gamma))
        return f, g, Kb
    return f, 0., Kb


def monster_img_log(P, Q, K, epsilon=0.01, gamma=1., maxiter=2000, tol=1e-7,
                    verbose=False, f0=0., compute_grad=False, Kb=None):
    """
    Compute the convolutional Sinkhorn-based Wasserstein divergence
    in the log-domain for 2D histograms, improving numerical stability.

    Parameters
    ----------
    P : torch.Tensor, shape (n_P, H, W) or (H, W)
        First batch or single input of 2D histograms (non-negative).
    Q : torch.Tensor, shape (n_Q, H, W) or (H, W)
        Second batch or single input of 2D histograms (non-negative).
    K : torch.Tensor, shape (H, W)
        Convolution kernel.
    epsilon : float, default=0.01
        Entropic regularization parameter.
    gamma : float, default=1.0
        KL-divergence weight.
    maxiter : int, default=2000
        Max number of Sinkhorn iterations.
    tol : float, default=1e-7
        Convergence threshold.
    verbose : bool, default=False
        If True, print warning on non-convergence.
    f0 : float, default=0.0
        Constant added to initial cost.
    compute_grad : bool, default=False
        Whether to compute gradients.
    Kb : torch.Tensor, optional
        Initial convolution estimate.

    Returns
    -------
    f : torch.Tensor, shape (n_P, n_Q)
        Wasserstein divergence values.
    g : torch.Tensor or float
        Gradients (if compute_grad), else 0.
    Kb : torch.Tensor
        Final marginal convolution.
    """
    if Q.ndimension() == 2:
        Q = Q[None, :, :]
    if P.ndimension() == 2:
        P = P[None, :, :]
    Qs = Q.sum(dim=(1, 2))
    Ps = P.sum(dim=(1, 2))
    f = gamma * (Qs[None, :] + Ps[:, None]) + epsilon * f0
    P = torch.log(P + 1e-100)
    Q = torch.log(Q + 1e-100)

    if compute_grad:
        g = torch.ones(len(P), *Q.shape, dtype=Q.dtype, device=Q.device)

    fi = gamma / (gamma + epsilon)

    log = {'cstr': [], 'obj': [], 'flag': 0, 'a': [], 'b': []}
    cstr = 10
    b = torch.zeros(len(Ps), *Q.shape, dtype=Q.dtype, device=Q.device)
    Kb = convol_huge_imgs_log(b, K)
    bold = b.clone()
    for i in range(maxiter):
        a = fi * (P[:, None, :, :] - Kb)
        Ka = convol_huge_imgs_log(a, K.t())
        b = fi * (Q[None, :, :, :] - Ka)
        Kb = convol_huge_imgs_log(b, K)
        with torch.no_grad():
            if torch.isnan(Kb).any():
                raise ValueError("Nan values found in Sinkhorn :(")
            if i % 10 == 0:
                cstr = abs(bold - b).max()
                cstr /= max(1, abs(bold).max(), abs(b).max())
            bold = b.clone()
            log["cstr"].append(cstr)

            if cstr < tol:
                break
    if i == maxiter - 1 and verbose:
        warnings.warn("*** Maxiter reached ! err = {} ***".format(cstr))
        log['flag'] = 3

    f += - (epsilon + 2 * gamma) * torch.exp(a + Kb).sum((2, 3))

    if compute_grad:
        g = gamma * (1 - torch.exp(- a * epsilon / gamma))
        return f, g, Kb
    return f, 0., Kb


def monster(P, Q, K, epsilon=0.01, gamma=1., maxiter=2000, tol=1e-7,
            verbose=False, f0=0., compute_grad=False, Kb=None):
    """
    Compute the Sinkhorn-based Wasserstein divergence between 1D histograms.

    Parameters
    ----------
    P : torch.Tensor, shape (n_P, d) or (d,)
        Source histograms (each column is a distribution).
        (!) It can be a nested tensor.
    Q : torch.Tensor, shape (n_Q, d) or (d,)
        Target histograms.
    K : torch.Tensor, shape (d, d)
        Kernel matrix computed from cost (e.g., exp(-C / epsilon)).
    epsilon : float, default=0.01
        Entropic regularization parameter.
    gamma : float, default=1.0
        KL-penalty weight.
    maxiter : int, default=2000
        Maximum Sinkhorn iterations.
    tol : float, default=1e-7
        Convergence threshold.
    verbose : bool, default=False
        Warn if maximum iteration is reached.
    f0 : float, default=0.0
        Additive constant to base divergence.
    compute_grad : bool, default=False
        Whether to return gradients.
    Kb : torch.Tensor, optional
        Right marginal kernel product (precomputed).

    Returns
    -------
    f : torch.Tensor, shape (n_P, n_Q)
        Divergence matrix.
    g : torch.Tensor or float
        Gradient of f w.r.t. P (if compute_grad=True), else 0.
    Kb : torch.Tensor
        Final transport kernel.
    """

    nested_mode = nested_test(P)

    if compute_grad:
        g = torch.ones(len(P), *Q.shape, dtype=Q.dtype, device=Q.device)
        g *= gamma * (1 - 10 ** (20 * epsilon / gamma))

    #* Transpose
    # Convert to padded tensor EARLY (if nested)
    Q = Q.t()
    P_sub = torch.nested.to_padded_tensor(P, padding=0.0) if nested_mode else P.t()
    P = nested_op(P, lambda x: x.t())
    K_t = nested_op(K, lambda x: x.t())

    # Calculate sum
    if Q.ndimension() == 1:
        Q = Q[:, None]
    if nested_mode and P.ndimension() == 1:
        P = P[:, None]

    Qs = nested_sum(Q,0)
    Ps = nested_sum(P_sub, 0)

    f = gamma * (Qs[None, :] + Ps[:, None]) + epsilon * f0
    # n is maximum of Ps, m is the timesteps
    n, m = f.shape
    idq = Qs > -1e-2
    idp = Ps > -1e-2

    # Handle nested case for masking
    if nested_test(P):
        Q = Q[:, idq]
        # For nested P, we need to mask each element individually
        masked_P = []
        for p in P.unbind():
            if p.shape[0] > len(idp):  # Handle size mismatch
                p = p[:len(idp)]
            masked_P.append(p[idp[:p.shape[0]]])
        P = nested_tensor(masked_P)
    else:
        Q = Q[:, idq]
        P = P[:, idp]

    fi = gamma / (gamma + epsilon)

    log = {'cstr': [], 'obj': [], 'flag': 0, 'a': [], 'b': []}
    cstr = 10

    if Kb is None:
        if nested_test(P):
            # Create nested tensor of ones matching P's structure
            b_shapes = [(p.shape[0], Q.shape[1]) for p in P.unbind()]
            b = nested_tensor([torch.ones(s, dtype=P.dtype, device=P.device) for s in b_shapes])
        else:
            b = torch.ones((*P.shape, Q.shape[1]), dtype=P.dtype, device=P.device)
        Kb = convol_huge(b, K)

    #?? if P is nested then K, a, b will all be nested tensors
    bold = clone(b)
    for i in range(maxiter):
        a = nested_op([P, Kb], lambda x: (x[0][:,:,None]/x[1])**fi)
        Ka = convol_huge(a, K_t)

        #? If K is a nested tensor, Ka is a nested tensor
        b = nested_op([Q, Ka], lambda x: (x[0][:,None,:]/x[1])**fi)
        Kb = convol_huge(b, K)

        with torch.no_grad():
            if i % 10 == 0:
                cstr = nested_max_abs_diff(bold, b)
                cstr /= max(1, nested_max(bold).abs(), nested_max(b).abs())
            bold = clone(b)
            log["cstr"].append(cstr)

            if cstr < tol:
                break
            if torch.isnan(torch.nested.to_padded_tensor(Kb,padding = 0.0)).any() if nested_test(Kb) else torch.isnan(Kb).any():
                warnings.warn("Numerical Errors ! Stopped at last stable "
                              "iteration.")
                break

    if i == maxiter - 1 and verbose:
        warnings.warn("*** Maxiter reached ! err = {} ***".format(cstr))
        log['flag'] = 3
    # Calculate plsum - handle both nested and regular cases
    plsum = nested_sum(a * Kb, 0)
    # if nested_test(a):
    #     # Nested tensor case
    #     plsum_components = []
    #     for a_t, Kb_t in zip(a.unbind(), Kb.unbind()):
    #         plsum_components.append((a_t * Kb_t).sum(0))
    #     plsum = nested_tensor(plsum_components)
    # else:
    #     # Regular tensor case
    #     plsum = (a * Kb).sum(0)

    # Loop through all the timestep
    for j in range(m):
        if idq[j]:
            f[idp, j] += - (epsilon + 2 * gamma) * (plsum[:, j] if not nested_test(plsum) else torch.stack([p[j] for p in plsum.unbind()]))
    if compute_grad:
        if nested_test(a):
        # Nested gradient computation
            g_components = []
            for a_t in a.unbind():
                g_t = torch.zeros_like(a_t)
                for j in range(m):
                    if idq[j]:
                        g_t[:, j] = gamma * (1 - a_t[:, j] ** (- epsilon / gamma))
                g_components.append(g_t)
            g = nested_tensor(g_components)
        else:
            # Regular gradient computation
            g = torch.zeros_like(a)
            for j in range(m):
                if idq[j]:
                    g[idp, j] = gamma * (1 - a[:, :, j].t() ** (- epsilon / gamma))
        return f, g, Kb
    return f, 0., Kb


def monster_log(P, Q, K, epsilon=0.01, gamma=1., maxiter=2000, tol=1e-7,
                verbose=False, f0=0., compute_grad=False, Kb=None):
    """
    Compute the log-domain Sinkhorn-based Wasserstein divergence
    between 1D histograms for increased numerical stability.

    Parameters
    ----------
    P : torch.Tensor, shape (n_P, d) or (d,)
        Source histograms.
    Q : torch.Tensor, shape (n_Q, d) or (d,)
        Target histograms.
    K : torch.Tensor, shape (d, d)
        Cost kernel matrix.
    epsilon : float, default=0.01
        Entropic regularization.
    gamma : float, default=1.0
        KL divergence penalty weight.
    maxiter : int, default=2000
        Max iterations for Sinkhorn updates.
    tol : float, default=1e-7
        Convergence threshold.
    verbose : bool, default=False
        Print warnings on max iterations.
    f0 : float, default=0.0
        Optional initialization constant.
    compute_grad : bool, default=False
        If True, compute gradients.
    Kb : torch.Tensor, optional
        Precomputed marginal kernel.

    Returns
    -------
    f : torch.Tensor, shape (n_P, n_Q)
        Pairwise divergence matrix.
    g : torch.Tensor or float
        Gradient values (if compute_grad), else 0.
    Kb : torch.Tensor
        Final kernel product.
    """

    if compute_grad:
        g = torch.ones(len(P), *Q.shape, dtype=Q.dtype, device=Q.device)
        g *= gamma * (1 - 10 ** (20 * epsilon / gamma))
    Q = Q.t()
    P = P.t()
    if Q.ndimension() == 1:
        Q = Q[:, None]
    if P.ndimension() == 1:
        P = P[:, None]
    Qs = Q.sum(dim=0)
    Ps = P.sum(dim=0)
    f = gamma * (Qs[None, :] + Ps[:, None]) + epsilon * f0
    n, m = f.shape
    idq = Qs > -1e-2
    idp = Ps > -1e-2

    Q = Q[:, idp]
    P = P[:, idq]

    fi = gamma / (gamma + epsilon)

    log = {'cstr': [], 'obj': [], 'flag': 0, 'a': [], 'b': []}
    cstr = 10
    b = torch.zeros_like((*P.shape, Q.shape[1]), requires_grad=False)

    if Kb is None:
        Kb = convol_huge_log(b, K)

    bold = b.clone()
    for i in range(maxiter):
        a = fi * (P[:, :, None] - Kb)
        Ka = convol_huge_log(a, K.t())
        b = fi * (Q[:, None, :] - Ka)
        Kb = convol_huge_log(b, K)

        with torch.no_grad():
            if i % 10 == 0:
                cstr = abs(bold - b).max()
                cstr /= max(1, abs(bold).max(), abs(b).max())
            bold = b.clone()
            log["cstr"].append(cstr)

            if cstr < tol:
                break

    if i == maxiter - 1 and verbose:
        warnings.warn("*** Maxiter reached ! err = {} ***".format(cstr))
        log['flag'] = 3
    plsum = np.exp(a + Kb).sum(0)

    for j in range(m):
        if idq[j]:
            f[idp, j] += - (epsilon + 2 * gamma) * plsum[:, j]
    if compute_grad:
        for j in range(m):
            if idq[j]:
                g[idp, j] = 1 - torch.exp(- epsilon * a[:, :, j] / gamma)
                g[idp, j] *= gamma
        return f, g, Kb
    return f, 0., Kb


def divergencekl(P, Q, K, epsilon=0.01, gamma=1., maxiter=2000, tol=1e-7,
                 verbose=False, f0=0., compute_grad=False, log=False,
                 Kb=None, axx=None, ayy=None):

    print(P.ndimension(), Q.ndimension())
    if log:
        if P.ndimension() == 3:
            wxy = monster_img_log
            wxx = negentropy_img_log
        else:
            wxy = monster_log
            wxx = negentropy_log
    else:
        if P.ndimension() == 3:
            wxy = monster_img
            wxx = negentropy_img
        else:
            wxy = monster
            wxx = negentropy

    fxy, gxy, Kb = wxy(P, Q, K, epsilon, gamma, maxiter, tol,
                       verbose, f0, compute_grad, Kb=Kb)
    fxx, gxx, axx = wxx(P, K, epsilon, gamma, maxiter, tol,
                        verbose, f0, compute_grad, a=axx)
    fyy, _, ayy = wxx(Q, K, epsilon, gamma, maxiter, tol,
                      verbose, f0, compute_grad=False, a=ayy)
    f = fxy - 0.5 * (fxx[:, None] + fyy[None, :])
    G = 0.
    if compute_grad:
        G = gxy - gxx[:, None]
    # del fxy, gxy, fxx, gxx, fyy
    return f, G, Kb, axx, ayy


def negentropy_img_log(P, K, epsilon=0.01, gamma=1., maxiter=100, tol=1e-7,
                       verbose=False, f0=0., compute_grad=False, a=None):
    """Compute the Negentropy term W(p, p) elementwise.

    Parameters
    ----------
    P: numpy array (n_hists, width, width)
        Must be non-negative.
    M: numpy array (width, width)
        One dimensional kernel.
        if None, taken as euclidean gram matrix over [1:n_features]
        normalized by its median.
    epsilon: float > 0.
        Entropy weight. (optional, default 5 / n_features)
    maxiter: int > 0.
        Maximum number of iterations of the Sinkhorn algorithm.
    tol: float >= 0.
        Precision threshold of the Sinkhorn algorithm.
        (optional, default 1e-10)
    returnlog: boolean.
        default False. if True, a list of errors is returned.
    returnmarginal: boolean.
        default False. if True, returns the transport marginal.

    Returns
    -------
    float.
    Wasserstein divergence between p and q.

    """
    if P.ndimension() == 2:
        P = P[None, :, :]
    if gamma == 0.:
        fi = 1.
    else:
        fi = gamma / (gamma + epsilon)
    Ps = P.sum((1, 2))
    P = torch.log(P + 1e-10)
    if a is None:
        a = torch.zeros_like(P, requires_grad=False)
    aold = a.clone()
    log = {'cstr': [], 'obj': [], 'flag': 0}
    cstr = 10
    # torch.set_grad_enabled(grad)
    for i in range(maxiter):
        Ka = convol_imgs_log(a, K)
        a = 0.5 * (a + fi * (P - Ka))
        with torch.no_grad():
            cstr = abs(a - aold).max() / max(1, a.max(), aold.max())
            aold = a.clone()
            log["cstr"].append(cstr)

            if cstr < tol:
                break
    if i == maxiter - 1 and verbose:
        warnings.warn("*** Maxiter reached ! err = {} ***".format(cstr))
        log['flag'] = 3
    if gamma:
        psum = torch.exp(a + Ka).sum((1, 2))
        f = - epsilon * psum
        f = f + gamma * (2 * Ps - 2 * psum)
    else:
        f = 2 * (P * a).sum((1, 2))
    f = f + epsilon * f0
    if compute_grad:
        grad = gamma * (1 - torch.exp(- a * epsilon / gamma))
        return f, grad, a
    return f, 0., a


def wimgkl_parallel(P, Q, K, epsilon=0.01, gamma=1., maxiter=2000, tol=1e-7,
                    verbose=False, f0=0., compute_grad=False, Kb=None):
    """Compute the Wasserstein divergence (p1, q1), (p2, q2) ...

    Parameters
    ----------
    P: numpy array (width, width)
        Must be non-negative.
    Q: numpy array (n_imgs, width, width)
        Must be non-negative.
    M: numpy array (width, width)
        One dimensional kernel.
        if None, taken as euclidean gram matrix over [1:n_features]
        normalized by its median.
    epsilon: float > 0.
        Entropy weight. (optional, default 5 / n_features)
    maxiter: int > 0.
        Maximum number of iterations of the Sinkhorn algorithm.
    tol: float >= 0.
        Precision threshold of the Sinkhorn algorithm.
        (optional, default 1e-10)
    returnlog: boolean.
        default False. if True, a list of errors is returned.
    returnmarginal: boolean.
        default False. if True, returns the transport marginal.

    Returns
    -------
    float.
    Wasserstein divergence between p and q.

    """
    if Q.ndimension() == 2:
        Q = Q[None, :, :]
    Qs = Q.sum(dim=(1, 2))
    Ps = P.sum(dim=(1, 2))
    f = gamma * (Qs + Ps)

    fi = gamma / (gamma + epsilon)

    log = {'cstr': [], 'obj': [], 'flag': 0, 'a': [], 'b': []}
    cstr = 10
    bold = torch.ones_like(Q, requires_grad=False)
    if Kb is None:
        b = bold.clone()
        Kb = convol_imgs(b, K)
    plsumold = torch.zeros(len(Q), dtype=Q.dtype, device=Q.device)
    for i in range(maxiter):
        a = (P / Kb) ** fi
        Ka = convol_imgs(a, K.t())
        b = (Q / Ka) ** fi
        Kb = convol_imgs(b, K)
        plsum = (a * Kb).sum((1, 2))
        with torch.no_grad():
            cstr = abs(plsumold - plsum).max() / max(1, plsumold.max(),
                                                     plsum.max())
            plsumold = plsum
            log["cstr"].append(cstr)

            if cstr < tol or torch.isnan(plsum).any():
                break
    if i == maxiter - 1 and verbose:
        warnings.warn("*** Maxiter reached ! err = {} ***".format(cstr))
        log['flag'] = 3
    f += epsilon * (f0 - plsum)
    f += - 2 * gamma * plsum
    if compute_grad:
        g = gamma * (1 - a ** (- epsilon / gamma))
        return f, g, Kb
    return f, 0., Kb


def wkl_parallel(P, Q, K, epsilon=0.01, gamma=1., maxiter=2000, tol=1e-7,
                 verbose=False, f0=0., compute_grad=False, Kb=None):
    """Compute the Wasserstein divergence (p1, q1), (p2, q2) ...

    Parameters
    ----------
    P: numpy array (n_hists, dimension)
        Must be non-negative.
    Q: numpy array (n_hists, dimension)
        Must be non-negative.
    M: numpy array (width, width)
        One dimensional kernel.
        if None, taken as euclidean gram matrix over [1:n_features]
        normalized by its median.
    epsilon: float > 0.
        Entropy weight. (optional, default 5 / n_features)
    maxiter: int > 0.
        Maximum number of iterations of the Sinkhorn algorithm.
    tol: float >= 0.
        Precision threshold of the Sinkhorn algorithm.
        (optional, default 1e-10)
    returnlog: boolean.
        default False. if True, a list of errors is returned.
    returnmarginal: boolean.
        default False. if True, returns the transport marginal.

    Returns
    -------
    float.
    Wasserstein divergence between p and q.

    """
    Q = Q.t()
    P = P.t()
    if Q.ndimension() == 1:
        Q = Q[:, None]
    if P.ndimension() == 1:
        P = P[:, None]

    Qs = Q.sum(dim=0)
    Ps = P.sum(dim=0)
    f = gamma * (Qs + Ps)

    fi = gamma / (gamma + epsilon)

    log = {'cstr': [], 'obj': [], 'flag': 0, 'a': [], 'b': []}
    cstr = 10
    bold = torch.ones_like(Q, requires_grad=False)
    if Kb is None:
        b = bold.clone()
        Kb = K.mm(b)
    log = {'cstr': [], 'obj': [], 'flag': 0, 'a': [], 'b': []}
    cstr = 10
    bold = torch.ones_like(Q, requires_grad=False)
    if Kb is None:
        Kb = K.mm(bold)

    for i in range(maxiter):
        a = (P / Kb) ** fi
        Ka = K.t().mm(a)
        b = (Q / Ka) ** fi
        Kb = K.mm(b)
        with torch.no_grad():
            if i % 10 == 0:
                cstr = abs(bold - b).max()
                cstr /= max(1, abs(bold).max(), abs(b).max())
            bold = b.clone()
            log["cstr"].append(cstr)

            if cstr < tol:
                break
    if i == maxiter - 1 and verbose:
        warnings.warn("*** Maxiter reached ! err = {} ***".format(cstr))
        log['flag'] = 3
    plsum = (a * Kb).sum(0)
    f += epsilon * (f0 - plsum)
    f += - 2 * gamma * plsum
    if compute_grad:
        g = gamma * (1 - a ** (- epsilon / gamma))
        return f, g, Kb
    return f, 0., Kb


def amarikl(P, Q, Qtild, K, epsilon=0.01, gamma=1., maxiter=2000, tol=1e-7,
            verbose=False, f0=0., compute_grad=False, log=False,
            Kb=None, normalize=True):
    if log:
        if P.ndimension() == 3:
            wxy = monster_img_log
            wsym = wimgkl_parallel
        else:
            wxy = monster
            wsym = wimgkl_parallel
    else:
        if P.ndimension() == 3:
            wxy = monster_img
            wsym = wimgkl_parallel
        else:
            wxy = monster
            wsym = wkl_parallel
    fxy, gxy, Kb = wxy(P, Qtild, K, epsilon, gamma, maxiter, tol,
                       verbose, f0, compute_grad, Kb=Kb)
    if normalize:
        fyky, _, _ = wsym(Q, Qtild, K, epsilon, gamma, maxiter, tol,
                          verbose, f0, compute_grad)
    else:
        fyky = torch.zeros(len(Q), dtype=P.dtype)
    G = 0.
    if compute_grad:
        G = gxy
    # del fxy, gxy, fxx, gxx, fyy
    return fxy, G, Kb, fyky
