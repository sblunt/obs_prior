import numpy as np
from orbitize.kepler import _calc_ecc_anom
import matplotlib.pyplot as plt

# obs prior test

# for this test, the actual values of M, R0, and sigma_x/sigma_y shouldn't matter-- they 
# just end up multiplying the prior prob by a constant value. 

# draw uniform samples of P and e, then rejection sample using eq 36a to get samples from the obs-based prior
num_samples = int(1e7)

P = np.random.uniform(0, 1e10, num_samples)
e = np.random.uniform(0, 1, num_samples)

P_min, P_max = np.min(P), np.max(P)
e_min, e_max = np.min(e), np.max(e)

def j_astro(P, e, manom=0.2):

    manom_arr = np.ones(len(e)) * manom
    E = _calc_ecc_anom(manom_arr, e)

    J_astro = -(P**(1/3)) * (
        2 * (e**2 - 2) * np.sin(E)  +
        e * (3 * manom + np.sin(2*E)) + 
        3 * manom * np.cos(E)
    ) / (
        6 * np.sqrt(1 - e**2)
    )

    return np.abs(J_astro)


# use rejection sampling
manom = 0.7
prior_prob = j_astro(P, e, manom=manom)
prior_prob = prior_prob / np.nanmax(prior_prob)

random_samples = np.random.uniform(0,1,num_samples)

accepted = np.where(prior_prob >= random_samples)[0]
print('acceptance rate: {:.2f}'.format(100 * len(accepted)/num_samples))

fig, ax = plt.subplots(2,1)
ax[0].hist(P[accepted], bins=50, color='grey', alpha=0.5)
ax[1].hist(e[accepted], bins=50, color='grey', alpha=0.5)
ax[0].set_xlabel('P [yr]')
ax[1].set_xlabel('ecc')
plt.tight_layout()
plt.savefig('obs_prob.png', dpi=250)

# convert obs-prior samples to X and Y
def X_and_Y(P, e, manom):

    sma = -(P**(1/3))

    try:
        E = _calc_ecc_anom(manom * np.ones(len(e)), e)
    except TypeError:
        E = _calc_ecc_anom(manom, e)
    X = sma * (np.cos(E) - e)
    Y = sma * (np.sqrt(1 - e**2)) * np.sin(E)

    return X, Y

# plot dist of X and Y (should be uniform)
fig, ax = plt.subplots(2,1)
X, Y = X_and_Y(P[accepted], e[accepted], manom)
ax[0].hist(X, bins=50, alpha=0.5, color='grey')
ax[1].hist(Y, bins=50, alpha=0.5, color='grey')
ax[0].set_xlabel('X')
ax[1].set_xlabel('Y')
Xminmin, Yminmin = X_and_Y(P_min, e_min, manom)
Xminmax, Yminmax = X_and_Y(P_min, e_max, manom)
Xmaxmin, Ymaxmin = X_and_Y(P_max, e_min, manom)
Xmaxmax, Ymaxmax = X_and_Y(P_max, e_max, manom)

colors=['k','hotpink','rebeccapurple','grey']
ls = ['-', '--', '-.', ':']
labels=[
    'e$_{{\mathrm{{min}}}}$, P$_{{\mathrm{{min}}}}$', 
    'e$_{{\mathrm{{min}}}}$, P$_{{\mathrm{{max}}}}$',
    'e$_{{\mathrm{{max}}}}$, P$_{{\mathrm{{min}}}}$',
    'e$_{{\mathrm{{max}}}}$, P$_{{\mathrm{{max}}}}$',
]
for i, val in enumerate([Xminmin, Xminmax, Xmaxmin, Xmaxmax]):
    ax[0].axvline(val, color=colors[i], label=labels[i], ls=ls[i])
for i, val in enumerate([Yminmin, Yminmax, Ymaxmin, Ymaxmax]):
    ax[1].axvline(val, color=colors[i], ls=ls[i])
ax[0].legend()

plt.tight_layout()
plt.savefig('obs_prob_XY.png', dpi=250)

plt.figure()
plt.hist2d(X,Y, bins=50)
plt.savefig('obs_prob_jointXY.png', dpi=250)

plt.figure()
delta_x = X + Y
plt.hist(delta_x, bins=50)
plt.savefig('XplusY.png', dpi=250)