# %%

# Import required modules 
import numpy as np 
import matplotlib.pyplot as plt 
  
# Meshgrid 
# x, y = np.meshgrid(np.linspace(-5, 5, 10),  
#                    np.linspace(-5, 5, 10)) 
  
# # Directional vectors 
# u = -y/np.sqrt(x**2 + y**2) 
# v = x/(x**2 + y**2) 
  
# # Plotting Vector Field with QUIVER 
# plt.quiver(x, y, u, v, color='g') 
# plt.title('Vector Field') 
  
# # Setting x, y boundary limits 
# plt.xlim(-7, 7) 
# plt.ylim(-7, 7) 
  
# # Show plot with grid 
# plt.grid() 
# plt.show() 


a1 = 0.7      # prey birth rate 
a2 = 0.007    # prey-predator-collision rate
a3 = 1        # predator death rate
a4 = 0.007


def f(x, y):
    v = x**2 +y**2
    return np.array([np.exp(-2*v)*(x+y), np.exp(-2*v)*(x-y)])

t = np.linspace(-1., 1., 15)
x, y = np.meshgrid(t, t)

dx, dy = f(x, y)

fig = plt.figure(figsize = (8, 6))
plt.quiver(x, y, dx, dy, color='g') 
plt.title('Vector Field') 
plt.grid() 
plt.show()


# %%
