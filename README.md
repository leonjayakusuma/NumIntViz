# NumIntViz: Interactive Numerical Integration Dashboard

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An interactive tool designed to visualize and analyze classical numerical integration techniques. This project demonstrates core Data Science competencies, including algorithmic implementation, error analysis, and interactive visualization.

## 🚀 Key Features

- **Interactive Dashboard**: Real-time visualization of integration areas using `ipywidgets` and `Matplotlib`.
- **Advanced Numerical Methods**:
  - Riemann Sums (Left, Right, Midpoint)
  - Trapezoidal Rule
  - Simpson's Rule
  - Gaussian Quadrature
- **Method Comparison Mode**: Overlay two methods to directly compare their geometric approximation strategies.
- **Convergence Analysis**: Dynamic log-log plots showing $O(1/n^p)$ error decay, with customizable ranges.
- **Auto-Play Animation**: Visualize the convergence process as the number of intervals ($n$) increases.

## 📊 Data Science Skills Demonstrated

- **Numerical Analysis & Calculus**: Implementation of quadrature rules and understanding of order-of-accuracy.
- **Object-Oriented Programming (OOP)**: Modular logic in `integrator.py` for extensibility and testing.
- **Data Visualization**: Creating intuitive, interactive UIs using the Jupyter ecosystem.
- **Algorithm Optimization**: Analyzing error scaling and convergence properties.

## 🛠️ Installation & Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/NumIntViz.git
   cd NumIntViz
   ```

2. **Set up the environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Explore the Dashboard**:
   Open `numerical_integration.ipynb` in your preferred Jupyter environment (VS Code, JupyterLab, etc.) and run the cells.

## 📖 Mathematical Context

Numerical integration approximates the definite integral:
$$\int_{a}^{b} f(x) dx \approx \sum_{i=1}^{n} w_i f(x_i)$$

This project explores the trade-offs between computational cost (number of intervals $n$) and accuracy (absolute error), visualizing how higher-order methods like Simpson's Rule outperform simpler Riemann sums for smooth functions.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
