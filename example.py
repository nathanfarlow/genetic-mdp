import numpy as np
from scipy.spatial.distance import cosine

from mdpsolver import solve_mdp


def generate_points_in_unit_ball(n_points=100, n_dim=3):
    points = []
    while len(points) < n_points:
        point = np.random.normal(size=n_dim)
        if np.linalg.norm(point) <= 1:
            points.append(point)
    return np.array(points)


points = generate_points_in_unit_ball()

# Find the 6 points that maximize the diversity problem
# Modify these parameters as necessary
result = solve_mdp(points, 6, cosine, n_gen=500, n_pop=120, verbose=True)

print(f'Score: {-result.F}')
print('Subset:')
subset = np.where(result.X == 1)[0]
for i in subset:
    print(points[i])
