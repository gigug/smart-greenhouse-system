# Smart Irrigation System 🌱
Final Project from the course "Cyber-Physical Systems" @ Data Science &amp; Scientific Computing, University of Trieste, year 2021/2022, written in Python.

## Introduction
Smart Irrigation System (SIS) is a system that manages the internal operations of a greenhouse: watering plants and adjusting the temperature.
It contains 2 PID controllers, to adjust the process variables according to the desired values.

## Usage
`src` contains the following files:
- `SIS.py`: implements the class `SIS`; the attributes of this class regulate the parameters of the simulation:
such as the dimensions of the greenhouse, the quantity of soil, the external temperature, the range of action of the sprinkler and the air conditioning system.
- `PID.py`: implements the controllers used by `SIS`; the user can set the internal parameters of both controllers.
`main.py` imports `src` elements and runs the simulation.
