# StrikingSpeed
Repository for my Master's Thesis at the University of Applied Sciences in Wr. Neustadt

This python file analyses striking velocity for strikes recorded with the PhyPhox App. To use it attach the mobile phone to the wrist or ankle and start a linear acceleration experiment. When using data from a different sensor, please prepare a .csv file as follows:

Column 0: Time Stamps in seconds
Column 1: Acceleration along local x-axis, in m/s2
Column 2: Acceleration along local y-axis, in m/s2
Column 3: Acceleration along local z-axis, in m/s2
Column 4: Absolute Acceleration as sqrt(x^2 + y^2 + z^2)

The basis for the implemented methods can be found in:

Izzo, R., Varde'i, C. H., Materazzo, P., Cejudo, A., & Giovannelli, M. (2022). Dynamic inertial analysis of the technical boxing gesture of Jab. Journal of Physical Education and Sport, 22(3), 661-671.
Kimm, D., & Thiel, D. V. (2015). Hand speed measurements in boxing. Procedia Engineering, 112, 502-506.
Pierratos, T., & Polatoglou, H. M. (2020). Utilizing the phyphox app for measuring kinematics variables with a smartphone. Physics Education, 55(2), 025019.
Socci, M., Varde’i, C. H., Giovannelli, M., Cejudo-Palomo, A., D'Elia, F., Cruciani, A., & Izzo, R. (2021). Definition of physical-dynamic parameters in circular kick in Muay Thai through latest generation inertial sensors with a critical review of the literature.
