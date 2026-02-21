import unittest
import numpy as np
from integrator import NumericalIntegrator

class TestNumericalIntegrator(unittest.TestCase):
    def setUp(self):
        self.integrator = NumericalIntegrator()

    def test_riemann_left(self):
        # f(x) = x, area from 0 to 1 should be approx 0.5 with high n
        f = lambda x: x
        approx, *_ = self.integrator.riemann_left(f, 0, 1, 1000)
        self.assertAlmostEqual(approx, 0.5, places=2)

    def test_trapezoid(self):
        # f(x) = x^2, area from 0 to 3 should be 9
        f = lambda x: x**2
        approx, *_ = self.integrator.trapezoid(f, 0, 3, 1000)
        self.assertAlmostEqual(approx, 9.0, places=4)

    def test_simpson(self):
        # Simpson's rule is exact for polynomials of degree 3 or less
        f = lambda x: x**3
        approx, *_ = self.integrator.simpson(f, 0, 2, 10)
        self.assertAlmostEqual(approx, 4.0, places=7)

    def test_gaussian_quadrature(self):
        # Exact for polynomials up to degree 2n-1
        f = lambda x: x**5
        # 3-point quadrature should be exact for x^5 (2*3-1 = 5)
        approx, *_ = self.integrator.gaussian_quadrature(f, 0, 1, 3)
        self.assertAlmostEqual(approx, 1/6, places=7)

    def test_parse_function(self):
        f, expr, f_int, expr_int = self.integrator.parse_function("x**2")
        self.assertEqual(float(f(2)), 4.0)
        self.assertAlmostEqual(self.integrator.get_true_area(f_int, 0, 3), 9.0)

if __name__ == "__main__":
    unittest.main()
