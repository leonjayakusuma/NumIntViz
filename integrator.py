import numpy as np
import sympy

class NumericalIntegrator:
    """
    A class to handle numerical integration calculations and data generation for plotting.
    """
    
    def __init__(self):
        self.x_sym = sympy.symbols('x')

    def parse_function(self, func_str):
        """
        Parses a sympy-style function string into a callable function and its integral.
        Example: 'x**2'
        """
        try:
            # Handle potential 'f(x) =' prefix
            if '=' in func_str:
                func_str = func_str.split('=', 1)[1].strip()
            
            expr = sympy.sympify(func_str)
            f_lambdified = sympy.lambdify(self.x_sym, expr, 'numpy')
            
            # Analytical integral
            expr_int = sympy.integrate(expr, self.x_sym)
            f_int_lambdified = sympy.lambdify(self.x_sym, expr_int, 'numpy')
            
            return f_lambdified, expr, f_int_lambdified, expr_int
        except Exception as e:
            raise ValueError(f"Error parsing function: {e}")

    def riemann_left(self, f, a, b, n):
        if n <= 0: return 0, None, None, None
        x = np.linspace(a, b, num=n+1)
        x_left = x[:-1]
        y_left = f(x_left)
        total = np.sum(y_left) * (b - a) / n
        return total, x, x_left, y_left

    def riemann_right(self, f, a, b, n):
        if n <= 0: return 0, None, None, None
        x = np.linspace(a, b, num=n+1)
        x_right = x[1:]
        y_right = f(x_right)
        total = np.sum(y_right) * (b - a) / n
        return total, x, x_right, y_right

    def riemann_mid(self, f, a, b, n):
        if n <= 0: return 0, None, None, None
        x = np.linspace(a, b, num=n+1)
        x_mid = (x[:-1] + x[1:]) / 2
        y_mid = f(x_mid)
        total = np.sum(y_mid) * (b - a) / n
        return total, x, x_mid, y_mid

    def trapezoid(self, f, a, b, n):
        if n <= 0: return 0, None, None
        x = np.linspace(a, b, num=n+1)
        y = f(x)
        h = (b - a) / n
        total = (h / 2) * (y[0] + 2 * np.sum(y[1:-1]) + y[-1])
        return total, x, y

    def simpson(self, f, a, b, n):
        # Simpson's rule usually requires an even number of intervals
        if n <= 0: return 0, None, None
        if n % 2 != 0:
            n += 1 # Ensure n is even for Simpson's
            
        x = np.linspace(a, b, num=n+1)
        y = f(x)
        h = (b - a) / n
        
        total = (h / 3) * (y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]) + y[-1])
        return total, x, y

    def get_true_area(self, f_int, a, b):
        return float(f_int(b) - f_int(a))

    def get_error_metrics(self, approx_area, true_area):
        abs_error = abs(approx_area - true_area)
        rel_error = (abs_error / abs(true_area)) * 100 if true_area != 0 else 0
        return abs_error, rel_error

    def get_convergence_data(self, f, a, b, f_int, ns, method_name):
        true_area = self.get_true_area(f_int, a, b)
        errors = []
        
        method_map = {
            'Riemann Left': self.riemann_left,
            'Riemann Right': self.riemann_right,
            'Riemann Mid': self.riemann_mid,
            'Trapezoidal': self.trapezoid,
            'Simpson': self.simpson
        }
        
        calc_method = method_map.get(method_name)
        if not calc_method:
            return [], []
            
        for n in ns:
            approx, *_ = calc_method(f, a, b, n)
            abs_err, _ = self.get_error_metrics(approx, true_area)
            # Use small constant to avoid log(0)
            errors.append(max(abs_err, 1e-18))
            
        return ns, errors

    def gaussian_quadrature(self, f, a, b, n):
        """
        N-point Gaussian Quadrature. 
        Note: n here refers to the number of points, not intervals.
        """
        if n <= 0: return 0, None, None
        
        # Use numpy's leggauss for weights and nodes on [-1, 1]
        nodes, weights = np.linalg.lapack.leggauss(n) if hasattr(np.linalg, 'lapack') else np.polynomial.legendre.leggauss(n)
        
        # Transform nodes and weights to [a, b]
        transformed_nodes = 0.5 * (nodes + 1) * (b - a) + a
        transformed_weights = 0.5 * weights * (b - a)
        
        total = np.sum(transformed_weights * f(transformed_nodes))
        return total, transformed_nodes, f(transformed_nodes)
