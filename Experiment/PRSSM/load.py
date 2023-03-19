import scipy.io

"""
The input data file flapping_wing_aerodynamics.mat includes the time history of:
• [Euler angles of the wing] - ds_pos:
    - stroke (phi): the wing position in the horizontal stroke plane. which for us is the motor angle
    - deviation (psi): the vertical wing deviation from the stroke plane
    - pitch (alpha): looks like the angle of attack

• [The kinematic variables derived from ds_pos] - ds_u_raw: (at the unit dist along the axis of rot from wing’s base)
    - |v|: v is the linear velocity of the wing (..)
    - AoA: geometric angle of attack
    - a^x: normal acceleration (..)
    - a^y: chord-wise acceleration (..)
    - a^z: span-wise acceleration (..)
    - alpha_dot: pitch velocity
    - alpha_ddot: pitch acceleration
    
• [Aerodynamic forces and moments] - ds_y_raw:
    - F^x: normal aerodynamic force
    - F^y: chord-wise aerodynamic force
    - M^x: normal aerodynamic moment on the wing
    - M^y: chord-wise aerodynamic moment on the wing
    - M^z: span-wise aerodynamic moment on the wing
    
• [The standardized versions of ds_u_raw and ds_y_raw] - ds_u and ds_y with the mean ds_mean_u and ds_mean_y and the
                                                         standard deviation "ds_std_u" and ds_std_y vectors. Only the
                                                          training data is considered when normalizing.
"""


def read_mat_file(mat_file_path: str):
    mat = scipy.io.loadmat(mat_file_path)
    return mat


if __name__ == '__main__':
    mat = read_mat_file('flapping_wing_aerodynamics.mat')
