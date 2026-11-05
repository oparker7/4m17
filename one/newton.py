import numpy as np


def line_search(x, delta_x, f, grad_f, alpha=0.1, beta=0.5, max_it=25):
    """
    Backtracking line search.

    Parameters
    ----------
    x : array_like, shape (n,)
        Current point.
    delta_x : array_like, shape (n,)
        Descent direction.
    f : callable
        Objective function; f(x) -> scalar.
    grad_f : callable
        Gradient; grad_f(x) -> array_like shape (n,).
    alpha : float,
        Sufficient decrease parameter (0 < alpha < 0.5).
    beta : float,
        Shrinkage factor (0 < beta < 1).
    max_it : int,
        Maximum number of backtracking steps.

    Returns
    -------
    t : float
        Step size
    """
    t = 1

    # Precompute some constants
    fx = f(x)
    grad_fx_delta_x = np.dot(grad_f(x), delta_x)

    # Check delta_x is a descent direction
    if grad_fx_delta_x >= 0:
        raise RuntimeError("delta_x isn't a descent direction!")

    # Backtracking loop
    for it in range(max_it):
        if f(x + t * delta_x) <= fx + alpha * t * grad_fx_delta_x:
            break
        t *= beta
    else:
        print("Warning: maximum line search iteration limit reached")

    return t


def newton(x_0, f, grad_f, hess_f, eps=1e-3, alpha=0.1, beta=0.5, max_it=50):
    """
    Newton's method for unconstrained minimisation with backtracking line search.

    Parameters
    ----------
    x_0 : array_like, shape (n,)
        Initial point for the iteration.
    f : callable
        Objective function; f(x) → float.
    grad_f : callable
        Gradient function; grad_f(x) → ndarray of shape (n,).
    hess_f : callable
        Hessian function; hess_f(x) → ndarray of shape (n, n).
    eps : float, optional
        Stopping tolerance based on the Newton decrement λ(x). The solver stops when λ(x)^2 / 2 ≤ eps.
    alpha : float, optional
        Backtracking line search sufficient decrease parameter (0 < alpha < 0.5).
    beta : float, optional
        Backtracking line search shrinkage factor (0 < beta < 1).
    max_it : int, optional
        Maximum number of Newton iterations.

    Returns
    -------
    x : ndarray, shape (n,)
        The estimated minimiser of f.
    it : int
        Number of Newton iterations performed.
    """
    x = x_0.copy()

    for it in range(0, max_it):

        # Precompute gradient and Hessian
        g = grad_f(x)
        H = hess_f(x)

        # Newton step
        delta_x = - np.linalg.solve(H, g)

        # Newton decrement
        lmbda_sq = np.dot(delta_x, H @ delta_x)

        if lmbda_sq / 2 <= eps:
            break

        t = line_search(x, delta_x, f, grad_f, alpha, beta)

        x += t * delta_x
    else:
        print("Warning: maximum Newton iteration limit reached")

    return x, it


if __name__ == "__main__":
    def f_0(x):
        """
        Objective function.
        """
        return np.exp(x[0]) + np.exp(x[1]) + 0.5*(x[0]**2 + x[1]**2)

    def grad_f_0(x):
        """
        Gradient of f_0.
        """
        return np.array([np.exp(x[0]) + x[0],
                        np.exp(x[1]) + x[1]])

    def hess_f_0(x):
        """
        Hessian of f_0.
        """
        return np.array([[np.exp(x[0]) + 1.0, 0.0],
                        [0.0, np.exp(x[1]) + 1.0]])

    # Starting point
    x0 = np.array([-1.5, 2.5])

    # Solve using Newton's method
    x_star, it = newton(x0, f_0, grad_f_0, hess_f_0, eps=1e-6)

    print(f"Initial point:      {x0}")
    print(f"Converged in:       {it} iterations")
    print(f"Optimal solution:   {x_star}")
    print(f"Optimal value:      {f_0(x_star)}")
