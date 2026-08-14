# First ML

# x = body mass
# y = bill length

# y-hat = wx + b

import pandas as pd
import random
import matplotlib.pyplot as plt # for when I want to visualize the graph

df = pd.read_csv("C:/Users/xandr/Desktop/Xandre/Programming/Python projects/penguins.csv")
# drops NaN values to make calculations easier
df = df[['body_mass_g', 'bill_length_mm']].dropna()

act_x = df['body_mass_g'].tolist()
act_y = df['bill_length_mm'].tolist()

x = act_x.copy()
y = act_y.copy()

error_container = []
MSE_record = []
w_history = []
b_history = []
MSE_dict = {}
l_mse_w = 0
b = 0
w = 0 # easier to understand what w is doing starting with 0

# for ensuring workability through output
# print(f"y-hat = {w:.3f} x {value} + {b}")
# print(f"{yhat:.2f}") 
def prediction():
    global w
    error_container = []
    for i, value in enumerate(x):
        yhat = w * value + b  
        # calculates error of guess
        error_v = y[i] - yhat
        error_container.append(error_v) # error
        
    MSE = sum(e**2 for e in error_container) / len(error_container)
    MSE_record.append(MSE)
    MSE_dict[MSE] = w

def optimization_w(w, i, w_amount):
        if MSE_record[i] < MSE_record[i-1]:
            w += w_amount 
        elif MSE_record[i] > MSE_record[i-1]: # is MSE increaasing (bad)?
            w -= w_amount
        else: # if mse is same as last one
            print(f'MSE has no change between {MSE_record[i]} and {MSE_record[i-1]}')
            w += w_amount # fallback
        return w

#
### Just notes, optional read:
#
# shit optimization i know. If mse increases the other won't know which variable made the error, 
# so I need to isolate the mse somehow it'll take too long to code an actual 
# good optimization loop for just a "understanding" project rather than an actual one
#

def optimization_b(b, i, b_amount):
        if MSE_record[i] < MSE_record[i-1]:
            b += b_amount 
        elif MSE_record[i] > MSE_record[i-1]: # is MSE increaasing (bad)?
            b -= b_amount
        else: # if mse is same as last one
            print(f'MSE has no change between {MSE_record[i]} and {MSE_record[i-1]}')
            b += b_amount # fallback
        return b


# optimization loop
for i in range(0, 100):
    print(f"=======  Loop Number: ({i + 1})  =======")
    prediction()
    w_history.append(w)
    b_history.append(b)

    # output
    if i < 1:
        print(f'MSE of Base Model: {MSE_record[i]:.2f}')
        print(f"The slope (should be 0) = {w:.4f}")
    else:
        print(f"MSE = {MSE_record[i]:.2f}")
        print(f"The slope = {w:.4f}")

    # mechanism
    if i < 1: # a first iteration safety thingy idk
        w += 0.001

    if i >= 1 and i < 4:
        w = optimization_w(w, i, 0.01)
        b = optimization_b(b, i, 0.1)

    if i >= 4:
        w = optimization_w(w, i, 0.001)
        b = optimization_b(b, i, 0.01)

    if i > 8:
        improvement = abs(MSE_record[i] - MSE_record[i-1])

        if improvement < 1e-5:
            print(
                f"Converged MSE = {MSE_record[i]:.6f}, "
                f"w = {w:.6f}"
            )
            break


lowest_mse_w = MSE_dict[min(MSE_dict)]

print("latest w:", w) 
print('latest MSE', MSE_record[-1])

print(f'w of lowest MSE so far: {lowest_mse_w}')
print(f'The lowest MSE so far: {min(MSE_dict)}')


# leaving project to begin studying Gradient Descent.