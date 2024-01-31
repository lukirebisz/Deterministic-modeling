# Deterministic-modeling

Project: biology

Let's imagine that a little kitten was born and the distribution of the pigment in its skin is random.

Cat owners are very interesting what the animal's fur will look like in the future.

In this project we have analyzed that problem using differential equations (Gierer-Meinhardt model: reaction-diffusion system of the activator-inhibitor type).
We created the numerical scheme for this model, checked its accuracy and then we performed simulations.

The simulations showed us how diverse patterns we can get depending on the parameters of the model. We have seen for example separate dots, joined or uniform patterns.

Last but not least, we have calculated the weighted variance of the final patterns to investigate how the diversity of patters depends on the parameters of Gierer-Meinhardt model.

The project was created using the Python language. We have programmed object-oriented.

The content of the files:

* main_code.py - the class Cat which includes all necessary methods to calculate the solutions of Gierer-Meinhardt model, show the results as the images or animations, calculate the mean weighted variance of the final patterns for the framework which is the image of the cat created from rectangles.

* Simulations.ipynb - simulations for the investigating model for the cat, examples of final patterns as the images and the animation, results for error of doubling the time step and for the mean weighted variance of final patterns.

* report.pdf - summary of results in the form of a report.