# COMP0216 2024-2025 Team 2 Project: Inverted Pendulum on a Cart

This repository implements multiple control strategies to balance an inverted pendulum on a cart. The project features full nonlinear dynamics, several controller designs (PID, LQR, and Nonlinear MPC), and an interactive graphical user interface (GUI) for simulation and visualization.

---

## Overview

**Objective:**  
Keep the inverted pendulum upright (i.e. \(\theta = 0\)) by applying appropriate horizontal forces to the cart.

**Approach:**  
- **Dynamics:**  
  The project uses the full nonlinear equations of motion (EoM) of the cart-pole system and its linearized version for controller design. The equations include cart dynamics and pendulum dynamics with damping.
  
- **Controllers:**  
  Multiple control strategies have been implemented:
  - **PID Controller:**  
    Uses proportional, integral, and derivative gains to minimize the pole angle error.
  - **LQR Controller:**  
    Designs an optimal state-feedback controller using a discretized linear model obtained from the nonlinear dynamics.
  - **Nonlinear MPC (NMPC):**  
    Solves a finite-horizon optimization problem using the full nonlinear dynamics with Euler integration and inequality constraints on the cart position and pole angle.
  
- **Simulation:**  
  The simulation employs two separate time steps:
  - **Controller update time step (dt_model):** Used for discretizing the dynamics for controllers such as LQR and NMPC.
  - **Integration time step (dt_sim):** Used for numerical integration of the continuous dynamics (Euler integration).

- **GUI:**  
  An interactive GUI (built with Matplotlib) allows users to:
  - Select the desired controller.
  - Tune controller parameters (e.g., PID gains or LQR Q/R matrices) through the slider bars.
  - Enable/disable external disturbance,sensor noise and filter.
  - Visualize time series of the cart position, pendulum angle, and control force.
  - See a real-time animation of the cart-pole system with a dynamic green arrow representing the applied control force.
  - Settling time shown at the end of simulation

---

## Repository Structure

- **source_cartpole.py:** 
-This module contains the core functionality of the project, including the implementation of various controller classes (such as PID, LQR, and NMPC), the nonlinear equations of motion for the cart-pole system, and a suite of utility functions (e.g., for discretization, filtering, and computing settling time).

- **2D_Main_GUI_Latest.py:**
- This is the main script that provides an interactive graphical user interface (GUI). It enables users to select and tune different controllers, run simulations of the cart-pole system, and visualize the results in real time, including animations and dynamic plots of system variables and control forces.


---

## Dependencies

- **Python 3.7+**
- [NumPy](https://numpy.org/)
- [SciPy](https://www.scipy.org/)
- [Matplotlib](https://matplotlib.org/)
- [cvxpy](https://www.cvxpy.org/) (optional; used for MPC formulations)
- Standard libraries: `math`, `typing`, `abc`, etc.

---

## Installation

-- Download file source_cartpole.py and file 2D_Main_GUI_Latest.py from Github

## Usage

-- 1.File Placement:  Ensure that both source_cartpole.py and 2D_Main_GUI_Latest.py are in the same directory.

-- 2. Running the GUI:  Launch the application by running the following command in your terminal: 
      python 2D_Main_GUI_Latest.py
-- 3. Selecting and Tuning Controllers:  
----- In the GUI, choose the controller you wish to use (PID, LQR, or NMPC) by selecting the appropriate radio button.

----- Adjust controller parameters using the provided sliders.

----- Optionally, you can enable disturbance and noise by checking the corresponding boxes and adjusting their amplitudes.

-- 4. Starting the Simulation:  Click the "Start" button to begin the simulation. The GUI will display real-time plots of the cart position, pendulum angle, control force, and an animation of the cart-pole system (including a visual arrow ----- representing the applied force).

-- 5. Switching Controllers/Settings:  To try a different controller or modify the settings (e.g., enable/disable disturbances or noise), click the "End" button to stop the current simulation.
----- Then, adjust your desired settings and press "Start" again to begin a new simulation.






