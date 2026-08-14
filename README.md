# first-ml-from-scratch
My first experiment building a linear regression model before learning gradient descent.

This project is my first attempt at building a machine learning system from scratch in Python and the math concepts I have recently learned.

I wanted to experiment with the underlying mechanics of linear regression and optimization before learning the formal methods used in machine learning.

# What the Project Does:
- Loads the Palmer Penguins dataset.
- Uses body mass as the input variable.
- Uses bill length as the target variable.
- Makes predictions using: ŷ = wx + b
- Calculates prediction errors.
- Calculates Mean Squared Error (MSE).
- Experiments with changing the slope (w) and bias (b).
- Uses MSE feedback to determine whether changes improved or worsened the model.
- Records parameter history for visualizing it through Matplotlib animation

At the time of writing, I had learned the mathematical foundations of:
- Functions and transformations
- Linear relationships
- Linear systems
- Correlation
- Linear regression
- Mean Squared Error

Instead of immediately using a standard optimization algorithm, I tried to design my own crude feedback system for adjusting the model's parameters.
This in-turn resulted in unreliable code, but the purpose of the project was to understand the problem before learning the formal solution.

# Next Step
The next stage will be learning and implementing gradient descent.
After understanding how the formal optimization works mechanically, I plan to compare the improved implementation with the original experimental/shitty version.
