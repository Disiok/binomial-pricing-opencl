import math

# Given
S0 = 50                     # Stock price  
t = 5.0 / 12.0              # Time  
v = 0.4                     # Volatility 
r = 0.1                     # Interest
X = 50                      # Strike 
n = 100                     # Iterations
optionType = -1              # Type of option: 1 for call, -1 for put

# ---------------------------Derived-------------------------------------------
dt = t / n                  # Time step
print "dt: ", dt

u = math.exp(v * dt ** 0.5) # Up factor
d = 1 / u                   # Down factor
print "u: ", u
print "d: ", d

disc = math.exp(r * dt)     # Discount factor
print "disc: ", disc

Pu = (disc - d) / (u -d)    # Up probability
Pd = 1 - Pu                 # Down probability
print "Pu: ", Pu
print "Pd: ", Pd

# Option values
V = [0] * n

# --------------------------Iterative calculation------------------------------
for i in range(n):
    Se = S0 * u ** (2 * i  - n)
    V[i] = max(0, optionType * (Se - X)) 

for j in range(n - 1, -1, -1):
    for i in range(j):
        V[i] = (Pd * V[i] + Pu * V[i + 1]) / disc

print V[0]
