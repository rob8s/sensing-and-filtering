import numpy as np
import matplotlib.pyplot as plt

WRITE = False # set to True if you want to write the data to a CSV file and process in another language

wall = [(0.1,4.), (3.,0.1)]

robot = [0.,0.]

def fit_line(data):
    # Calculate x and y coords
    x = data[:,0] * np.cos(data[:,1])
    y = data[:,0] * np.sin(data[:,1])
    # LSTSQ
    A = np.vstack([x, np.ones(len(x))]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return a, b

def noise_fit_line(data, noise=[0, 0.1, 0.2, 0.3, 0.4, 0.5], trials=50):
    r2_totals = []
    for sigma in noise:
        r2_tally = 0
        for _ in range(trials):
            # add noise to distance
            noise_data = data.copy()
            noise_data[:,0] += np.random.normal(0, sigma, len(data))
            # Fit new line
            a, b = fit_line(noise_data)

            # Tally R2 as we go
            r2_tally += r2(a, b, noise_data)
        r2_totals.append(r2_tally / trials)
    return r2_totals

def r2(a, b, data):
    # Solve for distance from robot to lien given thetas
    ab_dist = b / (np.sin(data[:,1]) - a * np.cos(data[:,1]))
    # Calculate r2
    ss_res = np.sum((data[:,0] - ab_dist)**2)
    ss_tot = np.sum((data[:,0] - np.mean(data[:,0]))**2)
    return 1 - (ss_res / ss_tot)


# plots results (will open a window / requires Xforwarding to view over SSH)
def PlotResults(data, a, b, wall, robot):
    fig, ax = plt.subplots()

    # visualize raw data
    remap_data = np.array([[d*np.cos(theta), d*np.sin(theta)] for (d, theta) in data])
    ax.scatter(remap_data[:,0], remap_data[:,1], color='red', marker='o', label="raw data")

    # visualize robot with heading at theta=0
    robot = plt.Circle(robot, 0.5, fill=False, edgecolor='purple')
    ax.add_patch(robot)
    heading = [[0, 0.5], [0,0]]
    ax.plot(heading[0], heading[1], color='purple')

    # visualize estimated wall line
    x = np.linspace(0,5,10)
    y = a*x+b
    ax.plot(x, y, color='blue', linestyle="--", label="estimated line")

    ax.legend()

    ax.set_xlim([-1, 5])
    ax.set_ylim([-1, 5])
    ax.axis('equal')
    plt.savefig('p1.png')

def FindDistances(wall, robot):
    wall = np.array(wall)
    robot = np.array(robot)
    data = []
    for theta in np.linspace(0, np.pi, 20):
        t, u, pt = ShootRay(robot, theta, wall[0], wall[1])
        if t > 0 and (0 < u) and (1 > u):
            dist = np.linalg.norm(pt - robot)
            data.append((round(dist,2), round(theta,2)))
    return np.array(data)

def ShootRay(pt, theta, v1, v2):
    '''
    # find line intersection parameter of edge (v1,v2)
    # https://stackoverflow.com/questions/14307158/how-do-you-check-for-intersection-between-a-line-segment-and-a-line-ray-emanatin/32146853
    # https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect/565282#565282
    '''
    def Cross2d(p,q):
        return p[0]*q[1] - p[1]*q[0]

    # points on ray are (x1,y1) + t*r
    r = (np.cos(theta), np.sin(theta))
    # points on segment are (x2,y2) + u*s
    s = v2-v1
    rXs = Cross2d(r,s)

    # if ray and target edge are parallel, will get divide by zero
    u = Cross2d(v1-pt, r)/rXs
    t = Cross2d(v1-pt, s)/rXs
    pint = np.array([pt[0] + np.cos(theta)*t, pt[1] + np.sin(theta)*t])
    return t, u, pint

def main():
    dat = FindDistances(wall, robot)
    print("raw data (distance, heading):", dat)
    # optionally, write data to file to process in another language
    if WRITE:
        np.savetxt('sensor-data.csv', dat, delimiter=',', fmt='%f')
    a, b = fit_line(dat)
    print(f"y = {a:.2f}x + {b:.2f}")
    PlotResults(dat, a, b, wall, robot)

    noise=[0, 0.1, 0.2, 0.3, 0.4, 0.5]
    r2_vals = noise_fit_line(dat, noise=noise,trials=50)
    print("R-squared values:", r2_vals)

    plt.figure()
    plt.plot(noise, r2_vals, marker='o')
    plt.xlabel('Noise (sigma)')
    plt.ylabel('Mean R-squared Value')
    plt.title('Goodness of fit vs Noise level')
    plt.grid(True)
    plt.savefig('p2.png')

if __name__ == '__main__':
    main()
