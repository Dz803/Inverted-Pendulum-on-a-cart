# COMP0216 2024-2025 Team 2 Project:Inverted Pendulum on a Cart 

This project implements multiple **control strategies** (PID, pole placement, LQR, etc.) to balance an inverted pendulum on a cart.

## Overview
- **Objective**: Keep the pendulum upright (`theta=0`) by applying horizontal forces.
- **Approach**: 
  1. Nonlinear equations of motion + linearization
  2. Various controllers (PID, continuous/discrete pole placement, LQR)
  3. Simulation in Python (`solve_ivp`), optional sensor noise, low-pass filtering

## Repository
├── 2D_main.py    # Main script (select controller) 
├── controller.py                   # Controller classes 
├── utils.py                        # Dynamics, discretization, plotting 
└── README.md                       # This file


## Dependencies
- Python ≥3.7  
- NumPy, SciPy, Matplotlib  
- (Optional) cvxpy for MPC  
- argparse 
- ABC
- typing

Install:
```bash
pip install numpy scipy matplotlib cvxpy 
```

## Usage

python 2D_main.py --controller {pid|dpole|dlqr} [--noise] [--animate]






