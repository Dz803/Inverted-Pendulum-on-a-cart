# Inverted Pendulum: Nonlinear Derivations & Linearized State-Space Matrices

This document summarizes the derivation of the nonlinear equations of motion using Lagrangian mechanics and their subsequent linearization.

---

## 1. Nonlinear Derivations

### 1.1 System Parameters

- \( M \): Cart mass  
- \( m \): Pendulum mass  
- \( l \): Pendulum length (to center of mass)  
- \( x \): Cart position  
- \( \theta \): Pendulum angle (from vertical)  
- \( F \): External force on the cart  
- \( b \): Cart damping coefficient  
- \( c \): Pendulum damping coefficient  

---

### 1.2 Kinetic Energy

**Cart:**
\[
T_{\text{cart}} = \frac{1}{2} M\,\dot{x}^2
\]

**Pendulum:**

Position of pendulum:
\[
x_p = x + l\,\sin\theta,\quad y_p = l\,\cos\theta
\]

Velocities:
\[
\dot{x}_p = \dot{x} + l\,\dot{\theta}\cos\theta,\quad \dot{y}_p = -l\,\dot{\theta}\sin\theta
\]

Kinetic energy:
\[
T_{\text{pendulum}} = \frac{1}{2} m \left[ \left(\dot{x} + l\,\dot{\theta}\cos\theta\right)^2 + \left(l\,\dot{\theta}\sin\theta\right)^2 \right]
\]
which simplifies to:
\[
T_{\text{pendulum}} = \frac{1}{2} m \left( \dot{x}^2 + 2l\,\dot{x}\,\dot{\theta}\cos\theta + l^2\,\dot{\theta}^2 \right)
\]

**Total Kinetic Energy:**
\[
T = \frac{1}{2}(M + m)\dot{x}^2 + m\,l\,\dot{x}\,\dot{\theta}\cos\theta + \frac{1}{2} m\,l^2\,\dot{\theta}^2
\]

---

### 1.3 Potential Energy

\[
V = m\,g\,l\,\cos\theta
\]

---

### 1.4 Lagrangian

\[
L = T - V = \frac{1}{2}(M + m)\dot{x}^2 + m\,l\,\dot{x}\,\dot{\theta}\cos\theta + \frac{1}{2} m\,l^2\,\dot{\theta}^2 - m\,g\,l\,\cos\theta
\]

---

### 1.5 Euler–Lagrange Equations

For a generalized coordinate \( q \) (either \( x \) or \( \theta \)):
\[
\frac{d}{dt}\left(\frac{\partial L}{\partial \dot{q}}\right) - \frac{\partial L}{\partial q} = Q,
\]
where the generalized forces are:
- \( Q_x = F - b\,\dot{x} \)
- \( Q_\theta = -c\,\dot{\theta} \)

---

### 1.6 Equations of Motion

**For \( x \):**
\[
\frac{\partial L}{\partial \dot{x}} = (M + m)\dot{x} + m\,l\,\dot{\theta}\cos\theta
\]
\[
\frac{d}{dt}\left(\frac{\partial L}{\partial \dot{x}}\right) = (M + m)\ddot{x} + m\,l\left(\ddot{\theta}\cos\theta - \dot{\theta}^2\sin\theta\right)
\]
Since \(\frac{\partial L}{\partial x} = 0\), the Euler–Lagrange equation gives:
\[
(M + m)\ddot{x} + m\,l\left(\ddot{\theta}\cos\theta - \dot{\theta}^2\sin\theta\right) = F - b\,\dot{x}
\]

**For \( \theta \):**
\[
\frac{\partial L}{\partial \dot{\theta}} = m\,l\,\dot{x}\cos\theta + m\,l^2\,\dot{\theta}
\]
\[
\frac{d}{dt}\left(\frac{\partial L}{\partial \dot{\theta}}\right) = m\,l\,\ddot{x}\cos\theta - m\,l\,\dot{x}\,\dot{\theta}\sin\theta + m\,l^2\,\ddot{\theta}
\]
\[
\frac{\partial L}{\partial \theta} = -m\,l\,\dot{x}\,\dot{\theta}\sin\theta + m\,g\,l\,\sin\theta
\]
Thus, the Euler–Lagrange equation becomes:
\[
m\,l\,\ddot{x}\cos\theta + m\,l^2\,\ddot{\theta} - m\,g\,l\,\sin\theta = -c\,\dot{\theta}
\]

---

### 1.7 Nonlinear State-Space Representation

Define the state vector:
\[
\mathbf{x} = \begin{bmatrix} x \\ \dot{x} \\ \theta \\ \dot{\theta} \end{bmatrix},\quad
\dot{\mathbf{x}} = \begin{bmatrix} \dot{x} \\ \ddot{x} \\ \dot{\theta} \\ \ddot{\theta} \end{bmatrix}.
\]

The equations of motion can be rearranged to express \(\ddot{x}\) and \(\ddot{\theta}\):

**Cart Equation:**
\[
(M + m)\ddot{x} + m\,l\,\ddot{\theta}\cos\theta - m\,l\,\dot{\theta}^2\sin\theta = F - b\,\dot{x}
\]

**Pendulum Equation:**
\[
m\,l\,\ddot{x}\cos\theta + m\,l^2\,\ddot{\theta} - m\,g\,l\,\sin\theta = -c\,\dot{\theta}
\]

After algebraic manipulation, the nonlinear state-space equations become:
\[
\ddot{x} = \frac{- m\,g\,l\,\sin\theta\cos\theta + m\,l\,\dot{\theta}^2\sin\theta + F - b\,\dot{x}}{M + m\,\sin^2\theta}
\]
\[
\ddot{\theta} = \frac{(M + m)g\,\sin\theta - m\,l\,\dot{\theta}^2\sin\theta\cos\theta + b\,\dot{x}\cos\theta - c\,\dot{\theta} - F\cos\theta}{l\,(M + m\,\sin^2\theta)}
\]

---

## 2. Linearized State-Space Matrices

### 2.1 Linearization Point

We linearize about the equilibrium:
\[
x = 0,\quad \dot{x} = 0,\quad \theta = 0,\quad \dot{\theta} = 0,\quad F = 0.
\]
Using the approximations:
\[
\sin\theta \approx \theta,\quad \cos\theta \approx 1,\quad \sin^2\theta \approx 0,\quad \dot{\theta}^2 \approx 0.
\]

---

### 2.2 Linearized Equations

The equations simplify to:
\[
\ddot{x} \approx \frac{-m\,g\,\theta - b\,\dot{x} + c\,\dot{\theta} + F}{M}
\]
\[
\ddot{\theta} \approx \frac{(M+m)g\,\theta + b\,\dot{x} - \frac{c(M+m)}{m\,l}\dot{\theta} - F}{l\,M}
\]

---

### 2.3 State-Space Form

Defining the state vector as before:
\[
\mathbf{x} = \begin{bmatrix} x \\ \dot{x} \\ \theta \\ \dot{\theta} \end{bmatrix},
\]
the linearized dynamics are:
\[
\dot{\mathbf{x}} = A_c\,\mathbf{x} + B_c\,F.
\]

---

### 2.4 Linearized Matrices

\[
A_c =
\begin{bmatrix}
0 & 1 & 0 & 0 \\
0 & -\frac{b}{M} & -\frac{m\,g}{M} & \frac{c}{M} \\
0 & 0 & 0 & 1 \\
0 & \frac{b}{l\,M} & \frac{(M+m)g}{l\,M} & -\frac{c(M+m)}{m\,l^2\,M}
\end{bmatrix},
\]
\[
B_c =
\begin{bmatrix}
0 \\
\frac{1}{M} \\
0 \\
-\frac{1}{l\,M}
\end{bmatrix}.
\]

---

This document presents both the derivation of the nonlinear equations using Lagrangian mechanics and the resulting linearized state-space matrices for the inverted pendulum system.
