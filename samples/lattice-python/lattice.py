import math

# Given
S0 = 100                    # Stock price  
t = 1.0                     # Time  
v = 0.3                     # Volatility 
r = 0.02                    # Interest
X = 100                     # Strike 
n = 500                     # Iterations
optionType = -1             # Type of option: 1 for call, -1 for put
isAmerican = True

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
V = [0] * (n + 1)

# --------------------------Iterative calculation------------------------------
for i in range(n + 1):
    Se = S0 * u ** i * d ** (n - i)
    V[i] = max(0, optionType * (Se - X)) 

for j in range(n - 1, -1, -1):
    for i in range(j + 1):
        V[i] = (Pd * V[i] + Pu * V[i + 1]) / disc
        if isAmerican:
            Si = S0 * u ** i * d ** (j - i)
            V[i] = max(V[i], max(0, optionType * (Si - X))) 
print V[0]
