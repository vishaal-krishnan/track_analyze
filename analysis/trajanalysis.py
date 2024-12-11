import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt
import math
import csv
import sys
import matplotlib.transforms as transforms
import matplotlib.colors as colors
import pandas as pd

from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Ellipse
from math import floor as fl
from random import randint
from scipy.interpolate import CubicSpline, interp1d, UnivariateSpline
from scipy.integrate import cumtrapz
from scipy.signal import find_peaks, butter, lfilter, hilbert, correlate, spectrogram
from scipy.fftpack import rfft, irfft, rfftfreq, fft, ifft, fftfreq
from matplotlib.widgets import Slider, Button
from scipy import stats, signal

from statsmodels.tsa.arima.model import ARIMA

try:
    from skimage.restoration import denoise_tv_chambolle
except ImportError:
    from skimage.filters import denoise_tv_chambolle


num = 4
type = 'CN/'
input_dir = 'data/cumulative/' + type + str(num) + '/'
output_dir = input_dir + 'processed/'


# Import data
with open(input_dir + 'r' + '.csv', newline='') as csvfile:
    r = np.loadtxt(csvfile,delimiter=',')

with open(input_dir + 'nest' + '.csv', newline='') as csvfile:
    nest = np.loadtxt(csvfile,delimiter=',')

with open(input_dir + 'feeder' + '.csv', newline='') as csvfile:
    feeder = np.loadtxt(csvfile,delimiter=',')

with open(input_dir + 'scale' + '.csv', newline='') as csvfile:
    scale = np.loadtxt(csvfile,delimiter=',')

with open(input_dir + 'phi' + '.csv', newline='') as csvfile:
    phi = np.loadtxt(csvfile,delimiter=',')

with open(input_dir + 'fps' + '.csv', newline='') as csvfile:
    fps = np.loadtxt(csvfile,delimiter=',')

with open(input_dir + 't' + '.csv', newline='') as csvfile:
    tau = np.loadtxt(csvfile,delimiter=',')

with open(input_dir + 't_drop' + '.csv', newline='') as csvfile:
    t_drop = np.loadtxt(csvfile,delimiter=',')

with open(input_dir + 't_change' + '.csv', newline='') as csvfile:
    t_change = np.loadtxt(csvfile,delimiter=',')


def plot_ellipse(ax, r_mean, sigma, eigenval, eigenvec, color):

    # Get the largest eigenvalue
    largest_eigenval = max(eigenval)

    # Get the index of the largest eigenvector
    largest_eigenvec_ind_c = np.argwhere(eigenval == max(eigenval))[0][0]
    largest_eigenvec = eigenvec[:,largest_eigenvec_ind_c]

    # Get the smallest eigenvector and eigenvalue
    smallest_eigenval = min(eigenval)
    if largest_eigenvec_ind_c == 0:
        smallest_eigenvec = eigenvec[:,1]
    else:
        smallest_eigenvec = eigenvec[:,0]

    # Calculate the angle between the x-axis and the largest eigenvector
    angle = np.arctan2(largest_eigenvec[1], largest_eigenvec[0]);

    # This angle is between -pi and pi.
    # Let's shift it such that the angle is between 0 and 2pi
    # if (angle < 0):
    #     angle = angle + 2*np.pi;

    pearson = sigma[0, 1]/np.sqrt(sigma[0, 0] * sigma[1, 1])

    # ell_radius_x = np.sqrt(1 + pearson)
    # ell_radius_y = np.sqrt(1 - pearson)
    ell_radius_x = largest_eigenval
    ell_radius_y = smallest_eigenval

    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor='none', edgecolor=color)

    # scale_x = np.sqrt(sigma[0, 0]*2.4477)
    scale_x = 10
    mean_x = r_mean[0]
    # scale_y = np.sqrt(sigma[1, 1]*2.4477)
    scale_y = 10
    mean_y = r_mean[1]

    transf = transforms.Affine2D().rotate(angle).scale(scale_x, scale_y).translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)

    return






# Rescale imported data
s1 = np.array(1920)
s2 = np.array(1080)
r[0,:] = r[0,:]*s1/s2
feeder[0] = feeder[0]*s1/s2

r = r*s2*scale
feeder = feeder*s2*scale

if type == 'T/':
    pass
else:
    nest[0] = nest[0]*s1/s2
    nest = nest*s2*scale


# Missing data imputation by smoothing spline interpolation
t = np.arange(0, tau[-1]-tau[0], 1/fps)
tau_sort_arg = np.argsort(tau - tau[0])
tau_sorted = np.sort(tau - tau[0])
r = r[:,tau_sort_arg]
r_data = r

tau_diff = np.ediff1d(np.insert(tau_sorted,0,0))

window = int(1*fps)
x_lin = interp1d(tau_sorted[np.where(tau_diff <= 1.5/fps)], r[0,np.where(tau_diff <= 1.5/fps)], kind='linear')
y_lin = interp1d(tau_sorted[np.where(tau_diff <= 1.5/fps)], r[1,np.where(tau_diff <= 1.5/fps)], kind='linear')
w_x = np.sqrt(np.reciprocal(1e-6 + np.convolve(np.ones(window)/window, np.ediff1d(np.squeeze(x_lin(t)),to_end=0,to_begin=0)**2, mode='same') - (np.convolve(np.ones(window)/window, np.ediff1d(np.squeeze(x_lin(t)),to_end=0,to_begin=0), mode='same'))**2 ))
w_x = w_x/np.max(w_x)
w_y = np.sqrt(np.reciprocal(1e-6 + np.convolve(np.ones(window)/window, np.ediff1d(np.squeeze(y_lin(t)),to_end=0,to_begin=0)**2, mode='same') - (np.convolve(np.ones(window)/window, np.ediff1d(np.squeeze(y_lin(t)),to_end=0,to_begin=0), mode='same'))**2 ))
w_y = w_y/np.max(w_y)

# r_lin = np.squeeze(np.array([x_lin(t),y_lin(t)]))

x_us = UnivariateSpline(tau_sorted[np.where(tau_diff <= 1.5/fps)], r[0,np.where(tau_diff <= 1.5/fps)], w=w_x[np.where(tau_diff <= 1.5/fps)], k=3, s=0.005*len(np.where(tau_diff <= 1.5/fps)) )
y_us = UnivariateSpline(tau_sorted[np.where(tau_diff <= 1.5/fps)], r[1,np.where(tau_diff <= 1.5/fps)], w=w_y[np.where(tau_diff <= 1.5/fps)], k=3, s=0.005*len(np.where(tau_diff <= 1.5/fps)) )

# plt.plot(w_x)
# print(y_us.get_residual())

vx_us = x_us.derivative(n=1)
vy_us = y_us.derivative(n=1)

ax_us = x_us.derivative(n=2)
ay_us = y_us.derivative(n=2)

r_us = np.squeeze(np.array([x_us(t),y_us(t)]))
v_us = np.squeeze(np.array([vx_us(t),vy_us(t)]))
a_us = np.squeeze(np.array([ax_us(t),ay_us(t)]))

r_pellet = r_us[:,np.min(np.where(t >= t_drop/fps-tau[0]))]

# Reference
r_ref = r_pellet
# r_ref = nest

len_scale = 1
d_scale = np.linalg.norm(feeder-r_ref)

nest = (nest - r_ref)/len_scale
feeder = (feeder - r_ref)/len_scale
r_us = (np.transpose(r_us) - r_ref)/len_scale
r_data = (np.transpose(r_data) - r_ref)/len_scale

rot_vec = feeder/np.linalg.norm(feeder)
rot_matrix = np.array([[-rot_vec[1], rot_vec[0]],[-rot_vec[0], -rot_vec[1]]])

r_us = np.matmul(rot_matrix, np.transpose(r_us))
r_data = np.matmul(rot_matrix, np.transpose(r_data))
feeder = np.matmul(rot_matrix, np.transpose(feeder))
nest = np.matmul(rot_matrix, np.transpose(nest))

if np.linalg.norm(feeder/d_scale-np.array([0,-1])) > 1e-6:
    print('error')
    sys.exit()

# Arclength parametrization
speed = np.linalg.norm(v_us,axis=0)
arc = cumtrapz(speed,t, initial=0)

x_s_us = UnivariateSpline(arc, x_us(t), s=0)
y_s_us = UnivariateSpline(arc, y_us(t), s=0)

# vx_s_us = x_s_us.derivative(n=1)
# vy_s_us = y_s_us.derivative(n=1)
# ax_s_us = x_s_us.derivative(n=2)
# ay_s_us = y_s_us.derivative(n=2)

x_us_temp = UnivariateSpline(tau_sorted[np.where(tau_diff <= 1.5/fps)], r_data[0,np.where(tau_diff <= 1.5/fps)], w=w_x[np.where(tau_diff <= 1.5/fps)], k=3, s=0.005*len(np.where(tau_diff <= 1.5/fps)) )
y_us_temp = UnivariateSpline(tau_sorted[np.where(tau_diff <= 1.5/fps)], r_data[1,np.where(tau_diff <= 1.5/fps)], w=w_y[np.where(tau_diff <= 1.5/fps)], k=3, s=0.005*len(np.where(tau_diff <= 1.5/fps)) )
x_s_us_temp = UnivariateSpline(arc, x_us_temp(t), s=0)
y_s_us_temp = UnivariateSpline(arc, y_us_temp(t), s=0)
vx_s_us = x_s_us_temp.derivative(n=1)
vy_s_us = y_s_us_temp.derivative(n=1)
ax_s_us = x_s_us_temp.derivative(n=2)
ay_s_us = y_s_us_temp.derivative(n=2)

s = np.arange(0, arc[-1]-arc[0], 0.01/len_scale)

v_s_us = np.sqrt(vx_s_us(s)**2+vy_s_us(s)**2)

r_s_us = np.squeeze(np.array([x_s_us(s),y_s_us(s)]))
r_s_us = (np.transpose(r_s_us) - r_ref)/len_scale
r_s_us = np.matmul(rot_matrix, np.transpose(r_s_us))

s_drop = arc[np.min(np.where(t >= t_drop/fps-tau[0]))]
s_change = arc[np.min(np.where(t >= t_change/fps-tau[0]))]

kappa_s = np.multiply(vx_s_us(s),ay_s_us(s)) - np.multiply(vy_s_us(s),ax_s_us(s))
kappa_s_hist = np.histogram(kappa_s)

scale_min = 1
scale_max = 50
kappa_cwt = signal.cwt(kappa_s-np.mean(kappa_s), signal.ricker, np.arange(scale_min, scale_max, 1, dtype=float))
kappa_cwt = np.flipud(kappa_cwt)

haar_lvl = pywt.dwt_max_level(kappa_s.shape[0], pywt.Wavelet('haar').dec_len)
kappa_haar_coeffs = pywt.wavedec(kappa_s-np.mean(kappa_s), 'haar', level=haar_lvl)

lvl_reconst = haar_lvl
kappa_haar = np.array([kappa_haar_coeffs[-1]])
for i in range(haar_lvl - 1):
    conc = np.array([np.repeat(kappa_haar_coeffs[haar_lvl - 1 - i], pow(2, i + 1))])
    kappa_haar = np.concatenate([kappa_haar, conc[:,0:kappa_haar.shape[1]]])

kappa_haar_coeffs_reconst = pywt.wavedec(kappa_s-np.mean(kappa_s), 'haar', level=haar_lvl)
kappa_s_scale = pywt.idwt(None, kappa_haar_coeffs_reconst[-lvl_reconst], 'haar', mode='symmetric')
# kappa_s_scale += np.mean(kappa_s)

kappa_reconst = np.zeros(s.shape[0])
# bandwidth = haar_lvl
scale_small = 4
bandwidth = haar_lvl - scale_small + 1

for i in range(bandwidth):
    l = scale_small + i
    kappa_reconst += pywt.upcoef('d', kappa_haar_coeffs[-l], 'haar',level=l)[:s.shape[0]]

kappa_reconst += pywt.upcoef('a', kappa_haar_coeffs[0], 'haar',level=haar_lvl)[:s.shape[0]]
kappa_reconst_cumsum = np.cumsum(kappa_reconst)

err = (np.mean(np.abs(kappa_s - kappa_reconst)**2))**0.5
# print(err)
# sys.exit()

autocorr_kappa = np.correlate(kappa_s -np.mean(kappa_s), kappa_s-np.mean(kappa_s),'full') / (kappa_s.shape[0]*np.std(kappa_s)**2)

# Variables of interest
kappa = np.divide( np.multiply(vx_us(t),ay_us(t)) - np.multiply(vy_us(t),ax_us(t)) , (vx_us(t)**2 + vy_us(t)**2)**1.5 , out=np.zeros_like(t), where= (vx_us(t)**2 + vy_us(t)**2)**0.5 >= 1e-4)


u_T = np.divide( np.multiply(vx_us(t),ax_us(t)) + np.multiply(vy_us(t),ay_us(t)) , (vx_us(t)**2 + vy_us(t)**2)**0.5 , out=np.zeros_like(t), where= (vx_us(t)**2 + vy_us(t)**2)**0.5 >=1e-4 )
u_N = np.divide( np.multiply(vx_us(t),ay_us(t)) - np.multiply(vy_us(t),ax_us(t)) , (vx_us(t)**2 + vy_us(t)**2)**0.5 , out=np.zeros_like(t), where= (vx_us(t)**2 + vy_us(t)**2)**0.5 >=1e-4 )

omega = np.divide( np.multiply(vx_us(t),ay_us(t)) - np.multiply(vy_us(t),ax_us(t)) , (vx_us(t)**2 + vy_us(t)**2) , out=np.zeros_like(t), where= (vx_us(t)**2 + vy_us(t)**2)**0.5 >=1e-4 )
theta = cumtrapz(omega,t, initial=0)

omega_s = np.multiply(vx_s_us(s),ay_s_us(s)) - np.multiply(vy_s_us(s),ax_s_us(s))
del_omega_s = np.ediff1d(np.insert(omega_s,0,0))
theta_s = cumtrapz(omega_s, s, initial=0)

del_theta_s = np.ediff1d(np.insert(theta_s,0,0))
del_theta_s = np.arctan2(np.sin(del_theta_s), np.cos(del_theta_s))
del_theta_s_df = pd.DataFrame(del_theta_s)
del_theta_s_ma = del_theta_s_df.rolling(window=50,center=True).mean().to_numpy()
del_theta_s_mstd = del_theta_s_df.rolling(window=50,center=True).std().to_numpy()

integral_theta_s = cumtrapz(theta_s, s, initial=theta_s[0])
int_mean_theta_s = np.divide(integral_theta_s[1:], s[1:])

s_start = s[np.where(np.absolute(theta_s[1:] - int_mean_theta_s)*180/np.pi > 50)[0][0]]
t_start = t[np.min(np.where(arc-arc[0] >= s_start))]

search_t_pre_drop = np.array([t[np.min(np.where(t >= t_drop/fps-tau[0]))] - t_start])
search_s_scale_pre_drop = np.array([s_drop - s_start])/d_scale
search_s_pre_drop = np.array([s_drop - s_start])

tangent_x_s_pre = np.cos(theta_s[np.squeeze(np.where( (s >= s_start) & (s < s_drop) ))])
tangent_y_s_pre = np.sin(theta_s[np.squeeze(np.where( (s >= s_start) & (s < s_drop) ))])

tangent_corr_pre = (correlate(tangent_x_s_pre, tangent_x_s_pre,'full') + correlate(tangent_y_s_pre, tangent_y_s_pre,'full'))
ndata_pre = np.concatenate((np.arange(1,tangent_x_s_pre.shape[0]+1), np.arange(tangent_x_s_pre.shape[0],1,-1)))
tangent_corr_pre = np.divide(tangent_corr_pre,ndata_pre)
tangent_corr_pre_envelope = np.abs(hilbert(tangent_corr_pre))

tangent_corr_pre_cwt = signal.cwt(tangent_corr_pre, signal.ricker, np.arange(scale_min, scale_max, 1, dtype=float))
tangent_corr_pre_cwt = np.flipud(tangent_corr_pre_cwt)

tangent_x_s_post = np.cos(theta_s[np.squeeze(np.where( s >= s_drop ))])
tangent_y_s_post = np.sin(theta_s[np.squeeze(np.where( s >= s_drop ))])

tangent_corr_post = (correlate(tangent_x_s_post, tangent_x_s_post,'full') + correlate(tangent_y_s_post, tangent_y_s_post,'full'))
ndata_post = np.concatenate((np.arange(1,tangent_x_s_post.shape[0]+1), np.arange(tangent_x_s_post.shape[0],1,-1)))
tangent_corr_post = np.divide(tangent_corr_post,ndata_post)

tangent_corr_pre_mat = np.outer(tangent_x_s_pre, tangent_x_s_pre) + np.outer(tangent_y_s_pre, tangent_y_s_pre)
tangent_corr_post_mat = np.outer(tangent_x_s_post, tangent_x_s_post) + np.outer(tangent_y_s_post, tangent_y_s_post)

n_window_pre = tangent_corr_pre_mat.shape[0]//2
n_window_post = tangent_corr_post_mat.shape[0]//2

tangent_corr_pre_timefreq = []
for k in range(n_window_pre-1):
    tangent_corr_pre_timefreq.append(tangent_corr_pre_mat[k][k:k+n_window_pre-1])

tangent_corr_pre_timefreq = np.array(tangent_corr_pre_timefreq)

tangent_corr_post_timefreq = []
for k in range(n_window_post-1):
    tangent_corr_post_timefreq.append(tangent_corr_post_mat[k][k:k+n_window_post-1])

tangent_corr_post_timefreq = np.array(tangent_corr_post_timefreq)



# fig, (ax1,ax2) = plt.subplots(1,2)
# fig.tight_layout()
# ax1.grid(False, which='both')
# ax1.axhline(y=0, color='k')
# ax1.axvline(x=0, color='k')
# ax1.set(xlabel="s",ylabel="del")
# # ax.axis([0, tangent_corr_post_timefreq.shape[0],0,tangent_corr_post_timefreq.shape[1]])
# ax1.imshow(np.transpose(tangent_corr_pre_timefreq), interpolation='nearest', cmap="gray")
# # fig.colorbar(ax1.pcolor(np.transpose(tangent_corr_pre_timefreq)))
# ax2.imshow(np.transpose(tangent_corr_post_timefreq), interpolation='nearest', cmap="gray")
# # plt.show()
# # sys.exit()

# # Spectrogram
# nseg = 50
# freq_spec, s_spec, del_theta_s_xx = signal.spectrogram(del_theta_s, nperseg=nseg,  noverlap=nseg-1)
# figspec, axspec = plt.subplots()
# axspec.pcolormesh(s_spec, freq_spec, del_theta_s_xx)
# axspec.set(xlabel='s', ylabel='freq')

# ARIMA
# model = ARIMA(del_theta_s, order=(50,1,50))
# model_fit = model.fit()
# print(model_fit.summary())
# residuals = DataFrame(model_fit.resid)
# residuals.plot()
# plt.show()
# residuals.plot(kind='kde')
# plt.show()
# print(residuals.describe())

# # 1. Pre drop save
# np.savetxt(input_dir + 'search_t_pre_drop' + '.csv', search_t_pre_drop, delimiter=',')
# np.savetxt(input_dir + 'search_s_scale_pre_drop' + '.csv', search_s_scale_pre_drop, delimiter=',')
# np.savetxt(input_dir + 'search_s_pre_drop' + '.csv', search_s_pre_drop, delimiter=',')
# print(search_t_pre_drop)
# print(search_s_pre_drop)
# print(search_s_scale_pre_drop)
# print(d_scale)
# sys.exit()

nest_T = np.divide( np.multiply(vx_us(t),nest[0]-r_us[0,:]) + np.multiply(vy_us(t),nest[1]-r_us[1,:]) , (vx_us(t)**2 + vy_us(t)**2)**0.5 , out=np.zeros_like(t), where= (vx_us(t)**2 + vy_us(t)**2)**0.5 >=1e-4 )
nest_N = np.divide( np.multiply(vx_us(t),nest[1]-r_us[1,:]) - np.multiply(vy_us(t),nest[0]-r_us[0,:]) , (vx_us(t)**2 + vy_us(t)**2)**0.5 , out=np.zeros_like(t), where= (vx_us(t)**2 + vy_us(t)**2)**0.5 >=1e-4 )

nest_T_s = np.multiply(vx_s_us(s),nest[0]-r_s_us[0,:]) + np.multiply(vy_s_us(s),nest[1]-r_s_us[1,:])
nest_N_s = np.multiply(vx_s_us(s),nest[1]-r_s_us[1,:]) - np.multiply(vy_s_us(s),nest[0]-r_s_us[0,:])

nest_s = np.transpose(np.array([nest_T_s, nest_N_s]))
theta_nest_s = np.arctan2(nest_N_s,nest_T_s)


r_pellet = r_us[:,np.min(np.where(t >= t_drop/fps-tau[0]))]
r_pellet_scale = r_pellet/d_scale

# # 2. Pellet save
# np.savetxt(input_dir + 'r_pellet' + '.csv', r_pellet, delimiter=',')
# np.savetxt(input_dir + 'r_pellet_scale' + '.csv', r_pellet_scale, delimiter=',')
# print(r_pellet_scale)
# sys.exit()

pellet_T = np.divide( np.multiply(vx_us(t),r_pellet[0]-r_us[0,:]) + np.multiply(vy_us(t),r_pellet[1]-r_us[1,:]) , (vx_us(t)**2 + vy_us(t)**2)**0.5 , out=np.zeros_like(t), where= (vx_us(t)**2 + vy_us(t)**2)**0.5 >=1e-4 )
pellet_N = np.divide( np.multiply(vx_us(t),r_pellet[1]-r_us[1,:]) - np.multiply(vy_us(t),r_pellet[0]-r_us[0,:]) , (vx_us(t)**2 + vy_us(t)**2)**0.5 , out=np.zeros_like(t), where= (vx_us(t)**2 + vy_us(t)**2)**0.5 >=1e-4 )

pellet_s_T = np.multiply(vx_s_us(s),r_pellet[0]-r_s_us[0,:]) + np.multiply(vy_s_us(s),r_pellet[1]-r_s_us[1,:])
pellet_s_N = np.multiply(vx_s_us(s),r_pellet[1]-r_s_us[1,:]) - np.multiply(vy_s_us(s),r_pellet[0]-r_s_us[0,:])
d_pellet_s = (np.abs(np.squeeze(pellet_s_T))**2 + np.abs(np.squeeze(pellet_s_N))**2)**(0.5)
theta_pellet_s = np.arctan2(pellet_s_N,pellet_s_T)
del_theta_pellet_s = np.ediff1d(np.insert(theta_pellet_s,0,0))

cross_pdt = np.cross(np.transpose(np.array([vx_s_us(s),vy_s_us(s)])), -np.transpose(r_s_us)/np.linalg.norm(np.transpose(r_s_us), axis=1)[:,np.newaxis])
# cross_pdt = np.cross(np.transpose(np.array([vx_s_us(s),vy_s_us(s)])), -np.transpose(r_s_us))

# dot_pdt = np.dot(np.transpose(np.array([vx_s_us(s),vy_s_us(s)])), -np.transpose(r_s_us)/np.linalg.norm(np.transpose(r_s_us), axis=1)[:,np.newaxis])

phi_s = np.arctan2(r_s_us[1,:],r_s_us[0,:])*180/np.pi
phi_pellet_s = np.arctan2(pellet_s_N, pellet_s_T)*180/np.pi

feeder_T = np.divide( np.multiply(vx_us(t),feeder[0]-r_us[0,:]) + np.multiply(vy_us(t),feeder[1]-r_us[1,:]) , (vx_us(t)**2 + vy_us(t)**2)**0.5 , out=np.zeros_like(t), where= (vx_us(t)**2 + vy_us(t)**2)**0.5 >=1e-4 )
feeder_N = np.divide( np.multiply(vx_us(t),feeder[1]-r_us[1,:]) - np.multiply(vy_us(t),feeder[0]-r_us[0,:]) , (vx_us(t)**2 + vy_us(t)**2)**0.5 , out=np.zeros_like(t), where= (vx_us(t)**2 + vy_us(t)**2)**0.5 >=1e-4 )

## pre-pellet drop
r_pre = np.squeeze(np.array([r_us[0,np.where(t < t_drop/fps-tau[0])], r_us[1,np.where(t < t_drop/fps-tau[0])]]))
v_pre = np.squeeze(np.array([v_us[0,np.where(t < t_drop/fps-tau[0])], v_us[1,np.where(t < t_drop/fps-tau[0])]]))

r_pre_mean = np.mean(r_pre[:,np.squeeze(np.where( (t>=t_change) & (t<t_drop/fps-tau[0]) ))], axis=1)
r_pre_dev = np.transpose(r_pre[:,np.squeeze(np.where( (t>=t_change) & (t<t_drop/fps-tau[0]) ))]) - r_pre_mean
sigma_pre = np.matmul( np.transpose(r_pre_dev), r_pre_dev )/r_pre_dev.shape[0]

## post-pellet drop
# Set origin
x_post = r_us[0,np.where(t >= t_drop/fps-tau[0])]
y_post = r_us[1,np.where(t >= t_drop/fps-tau[0])]
r_post = np.squeeze(np.array([x_post, y_post]))
v_post = np.squeeze(np.array([v_us[0,np.where(t >= t_drop/fps-tau[0])], v_us[1,np.where(t >= t_drop/fps-tau[0])]]))

r_post_mean = np.mean(r_post, axis=1)
r_post_dev = np.transpose(r_post) - r_post_mean
sigma_post = np.matmul( np.transpose(r_post_dev), r_post_dev )/r_post_dev.shape[0]

eigenval_pre, eigenvec_pre = np.linalg.eig(sigma_pre)
eigenval_post, eigenvec_post = np.linalg.eig(sigma_post)


# KDE
xmin = np.minimum( np.min(r_post[0,:]), np.minimum(np.min(r_pre[0,np.squeeze(np.where( (t>=t_start) & (t<t_drop/fps-tau[0]) ))]), nest[0]) )
xmax = np.maximum( np.max(r_post[0,:]), np.maximum(np.max(r_pre[0,np.squeeze(np.where( (t>=t_start) & (t<t_drop/fps-tau[0]) ))]), nest[0]) )
ymin = np.minimum( np.min(r_post[1,:]), np.minimum(np.min(r_pre[1,np.squeeze(np.where( (t>=t_start) & (t<t_drop/fps-tau[0]) ))]), nest[1]) )
ymax = np.maximum( np.max(r_post[1,:]), np.maximum(np.max(r_pre[1,np.squeeze(np.where( (t>=t_start) & (t<t_drop/fps-tau[0]) ))]), nest[1]) )

#
# xmin = np.min(r_post[0,:])
# xmax = np.max(r_post[0,:])
# ymin = np.min(r_post[1,:])
# ymax = np.max(r_post[1,:])

X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([X.ravel(), Y.ravel()])

values_pre = np.vstack([r_pre[0,np.squeeze(np.where( (t>=t_start) & (t<t_drop/fps-tau[0]) ))], r_pre[1,np.squeeze(np.where( (t>=t_start) & (t<t_drop/fps-tau[0]) ))]])
kernel_pre = stats.gaussian_kde(values_pre)
rho_pre = np.reshape(kernel_pre(positions).T, X.shape)

values_post = np.vstack([r_post[0,:], r_post[1,:]])
kernel_post = stats.gaussian_kde(values_post)
rho_post = np.reshape(kernel_post(positions).T, X.shape)

pre_mode = [X.ravel()[np.argmax(rho_pre.ravel())], Y.ravel()[np.argmax(rho_pre.ravel())]]
post_mode = [X.ravel()[np.argmax(rho_post.ravel())], Y.ravel()[np.argmax(rho_post.ravel())]]

pre_mode_scale = pre_mode/d_scale
post_mode_scale = post_mode/d_scale

# # 3. Save modes
np.savetxt(input_dir + 'pre_mode' + '.csv', pre_mode, delimiter=',')
np.savetxt(input_dir + 'post_mode' + '.csv', post_mode, delimiter=',')
np.savetxt(input_dir + 'pre_mode_scale' + '.csv', pre_mode_scale, delimiter=',')
np.savetxt(input_dir + 'post_mode_scale' + '.csv', post_mode_scale, delimiter=',')
# print(pre_mode)
# print(post_mode)
# sys.exit()


r_pre_seq = np.linalg.norm(np.transpose(np.transpose(r_s_us[:,np.squeeze(np.where((s>=s_start) & (s<s_drop)))]) - pre_mode), axis=0)
r_pre_int = cumtrapz(np.linalg.norm(np.transpose(np.transpose(r_s_us[:,np.squeeze(np.where((s>=s_start) & (s<s_drop)))]) - pre_mode), axis=0)**2, s[np.squeeze(np.where((s>=s_start) & (s<s_drop)))], initial=0)
r_pre_std = np.sqrt(np.divide(r_pre_int, s[np.squeeze(np.where((s>=s_start) & (s<s_drop)))], out=np.zeros_like(s[np.squeeze(np.where((s>=s_start) & (s<s_drop)))]), where= s[np.squeeze(np.where((s>=s_start) & (s<s_drop)))] >= 1e-9))

# r_pre_seq = np.linalg.norm(np.transpose(np.transpose(r_s_us[:,np.squeeze(np.where((s>=s_start) & (s<s_drop)))])), axis=0)
# r_pre_int = cumtrapz(np.linalg.norm(np.transpose(np.transpose(r_s_us[:,np.squeeze(np.where((s>=s_start) & (s<s_drop)))]) ), axis=0)**2, s[np.squeeze(np.where((s>=s_start) & (s<s_drop)))], initial=0)
# r_pre_std = np.sqrt(np.divide(r_pre_int, s[np.squeeze(np.where((s>=s_start) & (s<s_drop)))], out=np.zeros_like(s[np.squeeze(np.where((s>=s_start) & (s<s_drop)))]), where= s[np.squeeze(np.where((s>=s_start) & (s<s_drop)))] >= 1e-9))

r_pre_std_diff = np.mean(np.ediff1d(r_pre_std))/0.01

# r_post_seq = np.linalg.norm(np.transpose(np.transpose(r_s_us[:,np.squeeze(np.where(s>=s_drop))]) - post_mode), axis=0)
# r_post_int = cumtrapz(np.linalg.norm(np.transpose(np.transpose(r_s_us[:,np.squeeze(np.where(s>=s_drop))]) - post_mode), axis=0)**2, s[np.squeeze(np.where(s>=s_drop))], initial=0)
# r_post_std = np.sqrt(np.divide(r_post_int, s[np.squeeze(np.where(s>=s_drop))], out=np.zeros_like(s[np.squeeze(np.where(s>=s_drop))]), where= s[np.squeeze(np.where(s>=s_drop))] >= 1e-9))

r_post_seq = np.linalg.norm(np.transpose(np.transpose(r_s_us[:,np.squeeze(np.where(s>=s_drop))]) ), axis=0)
r_post_int = cumtrapz(np.linalg.norm(np.transpose(np.transpose(r_s_us[:,np.squeeze(np.where(s>=s_drop))]) ), axis=0)**2, s[np.squeeze(np.where(s>=s_drop))], initial=0)
r_post_std = np.sqrt(np.divide(r_post_int, s[np.squeeze(np.where(s>=s_drop))], out=np.zeros_like(s[np.squeeze(np.where(s>=s_drop))]), where= s[np.squeeze(np.where(s>=s_drop))] >= 1e-9))

r_post_std_diff = np.mean(np.ediff1d(r_post_std))/0.01

# np.savetxt(input_dir + 'r_pre_std_diff' + '.csv', np.array([r_pre_std_diff]), delimiter=',')
# np.savetxt(input_dir + 'r_post_std_diff' + '.csv', np.array([r_post_std_diff]), delimiter=',')
# sys.exit()

# # 4. Save std
np.savetxt(input_dir + 'r_pre_std' + '.csv', np.array([s[np.where((s>=s_start) & (s<s_drop))]-s_start, r_pre_std]), delimiter=',')
np.savetxt(input_dir + 'r_post_std' + '.csv', np.array([s[np.where(s >= s_drop)]-s_start, r_post_std]), delimiter=',')
#
# omega_model = np.zeros(s.shape[0])
# f = np.divide(np.sin(theta_pellet_s),np.linalg.norm(np.transpose(r_s_us), axis=1))
# # f = np.sin(theta_pellet_s)
# for i in range(s.shape[0]-1):
#     omega_model[i+1] = omega_model[i] + f[i]
#
# fig_temp, ax_temp = plt.subplots()
# fig_temp.tight_layout()
# ax_temp.grid(True, which='both')
# ax_temp.axhline(y=0, color='k')
# ax_temp.axvline(x=0, color='k')
# # ax_temp.set(xlabel="$\sin(\\theta)$",ylabel="$\\omega$")
# # ax_temp.plot(s, -omega_s*(s[1]-s[0]) + np.divide(np.sin(theta_pellet_s),np.linalg.norm(np.transpose(r_s_us), axis=1))*(s[1]-s[0]), 'r')
# # ax_temp.plot(s, cross_pdt, 'b')
# ax_temp.plot(s, np.divide(omega_s*(s[1]-s[0]), np.sin(theta_pellet_s)), 'b')
# # ax_temp.plot(s[np.squeeze(np.where((s>=0.1*s_drop) & (s<0.5*s_drop)))], np.sin(theta_pellet_s[np.squeeze(np.where((s>=0.1*s_drop) & (s<0.5*s_drop)))]), 'r')
# # ax_temp.plot(s, np.divide(np.sin(theta_pellet_s),np.linalg.norm(np.transpose(r_s_us), axis=1))*(s[1]-s[0]), 'r')
# # ax_temp.plot(s, omega_s*(s0.1*[1]-s[0]),'r')
# # ax_temp.plot(s, omega_model)
# # ax_temp.scatter(np.divide(np.sin(theta_pellet_s),np.linalg.norm(np.transpose(r_s_us), axis=1)), del_theta_pellet_s)
# # ax_temp.scatter(omega_model,omega_s)
# # ax_temp.plot(np.sin(theta_pellet_s)[np.squeeze(np.where((s>=0.9*s_drop) & (s<s_drop)))], omega_s[np.squeeze(np.where((s>=0.9*s_drop) & (s<s_drop)))])

# Plots
fig1, track = plt.subplots()
# fig1.tight_layout()
# track.grid(True, which='both')
# track.axhline(y=0, color='k')
# track.axvline(x=0, color='k')
# # track.set(xlabel="X",ylabel="Y")
# track.set_autoscale_on(False)
# track.set_aspect('equal')
# track.axis([xmin,xmax,ymin,ymax])
#
# track.scatter(r_data[0,:],r_data[1,:], s=1, c='k', zorder=0, label='data')
# track.scatter(r_pellet[0],r_pellet[1], s=100, c='k', marker='o', zorder=3, label='dropped pellet')
# track.scatter(nest[0], nest[1] , s=100, c='g', marker='*', zorder=3, label='true nest location')
# track.plot([0, feeder[0]], [0, feeder[1]] , c='k', zorder=3)
#
# # track.scatter(r_data[0,np.min(np.where(arc-arc[0] >= s_start))],r_data[1,np.min(np.where(arc-arc[0] >= s_start))], s=100, c='k', marker='+', zorder=4, label='data')
#
# track.plot(r_post[0,:], r_post[1,:] , 'b', zorder=1, label='search post-pellet drop')
# track.plot(r_pre[0,np.squeeze(np.where(t<t_drop/fps-tau[0]))], r_pre[1,np.squeeze(np.where(t<t_drop/fps-tau[0]))], 'r', zorder=2, label='search pre-pellet drop')
# track.legend(loc='lower right', fontsize='large')

# track.quiver(r_s_us[0,:], r_s_us[1,:], vx_s_us(s), vy_s_us(s), zorder=3)

# track.set(xlabel="$\kappa$",ylabel="$|v|$")
# # track.scatter(np.abs(kappa_s[np.squeeze(np.where(s<s_drop))]), v_s_us[[np.squeeze(np.where(s<s_drop))]], s=100, c='r', marker='+')
# track.scatter(np.abs(kappa), np.linalg.norm(v_us,axis=0), s=50, c='r', marker='.',zorder=3)
# plt.xscale("log")
# plt.yscale("log")

# bgrnd = track.imshow(np.rot90(rho_post), cmap=plt.cm.gist_earth_r, extent=[xmin, xmax, ymin, ymax], zorder=0)
bgrnd = track.imshow(np.rot90(rho_pre), cmap=plt.cm.gist_earth_r, extent=[xmin, xmax, ymin, ymax], zorder=0)
# fgrnd_1 = track.scatter(r_post[0,int(fps)*k+3], r_post[1,int(fps)*k+3] , s=2, c='k', zorder=2)
fgrnd_2 = track.scatter(nest[0], nest[1] , s=75, c='r', marker='+', zorder=1, label='location of burrow')
fgrnd_3 = track.scatter(0,0, s=100, c='orange', marker='o', zorder=1, label = 'pellet drop location')
fig1.colorbar(bgrnd, ax=track)
track.legend(loc='upper right', fontsize='large')
plt.rc('font', family='Times New Roman')

plt.show()
sys.exit()


# fig82, ax82 = plt.subplots()
# fig82.tight_layout()
# ax82.grid(True, which='both')
# ax82.axhline(y=0, color='k')
# ax82.axvline(x=0, color='k')
# ax82.set(xlabel="s",ylabel="Distance to mode (m)")
# ax82.plot(s[np.where((s>=s_start) & (s<s_drop))], r_pre_seq, 'r')
# ax82.plot(s[np.where(s >= s_drop)], r_post_seq, 'b')
# # plt.xscale("log")
# # plt.yscale("log")


# # track.imshow(np.rot90(rho_pre), cmap=plt.cm.gist_earth_r,extent=[xmin, xmax, ymin, ymax], zorder=0)
# # track.imshow(np.rot90(rho_post), cmap=plt.cm.gist_earth_r, extent=[xmin, xmax, ymin, ymax], zorder=0)
#
# track.scatter(r_pre_mean[0], r_pre_mean[1] , s=30, c='g', marker='o')
# track.scatter(r_post_mean[0], r_post_mean[1] , s=30, c='black', marker='o')
#
# plot_ellipse(track, r_pre_mean-r_us[:,np.min(np.where(t >= t_drop/fps-tau[0]))], sigma_pre, eigenval_pre, eigenvec_pre, (1, 0, 0.2))
# plot_ellipse(track, r_post_mean, sigma_post, eigenval_post, eigenvec_post, (1, 0, 1))


# # Animate
# for k in range(r_post.shape[1]):
#     plt.cla()
#     if k>2:
#         values_post_iter = np.vstack([r_post[0,0:k], r_post[1,0:k]])
#         kernel_post_iter = stats.gaussian_kde(values_post_iter)
#         rho_post_iter = np.reshape(kernel_post_iter(positions).T, X.shape)
#         track.imshow(np.rot90(rho_post_iter[k-3]), cmap=plt.cm.gist_earth_r, extent=[xmin, xmax, ymin, ymax], zorder=0)
#         track.scatter(r_post[0,k], r_post[1,k] , s=2, c='k', zorder=2)
#         track.scatter(nest[0], nest[1] , s=50, c='r', marker='+', zorder=1)
#         plt.pause(1e-6)


# # save as gif
# def animate(k):
#     track.clear()
#
#     values_post_iter = np.vstack([r_post[0,0:int(fps)*k+3], r_post[1,0:int(fps)*k+3]])
#     kernel_post_iter = stats.gaussian_kde(values_post_iter)
#     rho_post_iter = np.reshape(kernel_post_iter(positions).T, X.shape)
#
#     bgrnd = track.imshow(np.rot90(rho_post_iter), cmap=plt.cm.gist_earth_r, extent=[xmin, xmax, ymin, ymax], zorder=0)
#     fgrnd_1 = track.scatter(r_post[0,int(fps)*k+3], r_post[1,int(fps)*k+3] , s=2, c='k', zorder=2)
#     # fgrnd_2 = track.scatter(nest[0], nest[1] , s=50, c='r', marker='+', zorder=1)
#     fgrnd_3 = track.scatter(0,0, s=100, c='b', marker='o', zorder=1)
#     return bgrnd,fgrnd_1,fgrnd_3
#
# ani = FuncAnimation(fig1, animate, interval=100, blit=True, repeat=True, frames=int(r_post.shape[1]/fps))
# ani.save(input_dir + "search.gif", dpi=300, writer=PillowWriter(fps=10))





# fig2, track2 = plt.subplots()
# fig2.tight_layout()
# track2.grid(True, which='both')
# track2.axhline(y=0, color='k')
# track2.axvline(x=0, color='k')
# track2.set(xlabel="x",ylabel="y")
# track2.set_autoscale_on(False)
# track2.plot(r_pre[0,:], r_pre[1,:] , 'b')
# track2.axis([ np.minimum( np.min(r_pre[0,:]), nest[0] ), np.maximum( np.max(r_pre[0,:]), nest[0] ), np.minimum( np.min(r_pre[1,:]), nest[1] ), np.maximum( np.max(r_pre[1,:]), nest[1] )])
# track2.scatter(nest[0], nest[1] , s=30, c='r', marker='x')
# track2.scatter(r_pre_mean[0], r_pre_mean[1] , s=30, c='g', marker='o')

# fig3, ax3 = plt.subplots()
# fig3.tight_layout()
# ax3.grid(True, which='both')
# ax3.axhline(y=0, color='k')
# ax3.axvline(x=0, color='k')
# ax3.set(xlabel="t",ylabel="|r|")
# ax3.plot(t[np.squeeze(np.where( (t>=t_change) & (t<t_drop/fps-tau[0]) ))], (np.abs(r_pre[0,np.squeeze(np.where( (t>=t_change) & (t<t_drop/fps-tau[0]) ))]-r_pellet[0])**2 + np.abs(r_pre[1,np.squeeze(np.where( (t>=t_change) & (t<t_drop/fps-tau[0]) ))]-r_pellet[1])**2)**(0.5), 'r')
# ax3.plot(t[np.where(t >= t_drop/fps-tau[0])], (np.abs(np.squeeze(x_post)-r_pellet[0])**2 + np.abs(np.squeeze(y_post)-r_pellet[1])**2)**(0.5), 'b')

# fig4, (ax41,ax42) = plt.subplots(2,1)
# fig4.tight_layout()
# ax41.grid(True, which='both')
# ax41.axhline(y=0, color='k')
# ax41.axvline(x=0, color='k')
# ax41.set(xlabel="t",ylabel="$v_x$")
# ax41.plot(t[np.where(t < t_drop/fps-tau[0])], np.squeeze(v_us[0,np.where(t < t_drop/fps-tau[0])]) , 'r')
# ax41.plot(t[np.where(t >= t_drop/fps-tau[0])], np.squeeze(v_us[0,np.where(t >= t_drop/fps-tau[0])]) , 'b')
# ax42.grid(True, which='both')
# ax42.axhline(y=0, color='k')
# ax42.axvline(x=0, color='k')
# ax42.set(xlabel="t",ylabel="$v_y$")
# ax42.plot(t[np.where(t < t_drop/fps-tau[0])], np.squeeze(v_us[1,np.where(t < t_drop/fps-tau[0])]), 'r')
# ax42.plot(t[np.where(t >= t_drop/fps-tau[0])], np.squeeze(v_us[1,np.where(t >= t_drop/fps-tau[0])]), 'b')


# v_x = np.squeeze(v_us[0,:])
# v_x_mean = np.mean(v_x)
# v_x = v_x - v_x_mean
# v_y = np.squeeze(v_us[1,:])
# v_y_mean = np.mean(v_y)
# v_y = v_y - v_y_mean

# filter_order = 2
# scale_ratio = 25
# cutoff_freq = fps/scale_ratio
# normalized_cutoff_freq = 2 * cutoff_freq / fps
# numerator_coeffs, denominator_coeffs = butter(filter_order, normalized_cutoff_freq)
# v_x = np.array(lfilter(numerator_coeffs, denominator_coeffs, v_x))
# v_y = np.array(lfilter(numerator_coeffs, denominator_coeffs, v_y))


# fig100, ax100 = plt.subplots()
# fig100.tight_layout()
# ax100.grid(True, which='both')
# ax100.axhline(y=0, color='k')
# ax100.axvline(x=0, color='k')
# ax100.set(xlabel="$v_x$",ylabel="$v_y$")
# ax100.set_autoscale_on(False)
# ax100.set_aspect('equal')
#
# for k in range(int(np.array(v_x).shape[0]/5)):
#     plt.cla()
#     ax100.axis([np.min(v_x),np.max(v_x),np.min(v_y),np.max(v_y)])
#     ax100.plot(v_x[range(0,int(5*k),5)], v_y[range(0,int(5*k),5)] , c='b')
#     plt.pause(1e-10)


# (freq, spec_density) = signal.welch(kappa[np.where(t >= t_drop/fps-tau[0])]-np.mean(kappa[np.where(t >= t_drop/fps-tau[0])]), nperseg=1024)
# fig110, ax110 = plt.subplots()
# fig110.tight_layout()
# ax110.grid(True, which='both')
# ax110.axhline(y=0, color='k')
# ax110.axvline(x=0, color='k')
# ax110.set(xlabel="normalized_frequency",ylabel="$P_{\kappa}$")
# ax110.plot(freq[np.squeeze(np.where(freq<=5))], np.abs(spec_density[np.squeeze(np.where(freq <=5))]), 'r')


# (freq, spec_density) = signal.welch(kappa_s[np.where(s >= s_drop)]-np.mean(kappa_s[np.where(s >= s_drop)]), nperseg=1024)
# fig11, ax11 = plt.subplots()
# fig11.tight_layout()
# ax11.grid(True, which='both')
# ax11.axhline(y=0, color='k')
# ax11.axvline(x=0, color='k')
# ax11.set(xlabel="normalized_frequency",ylabel="$P_{\kappa_s}$")
# ax11.plot(freq[np.squeeze(np.where(freq<=5))], spec_density[np.squeeze(np.where(freq <=5))])

# fig5, (ax5,ax6) = plt.subplots(2,1)
# fig5.tight_layout()
# ax5.grid(True, which='both')
# ax5.axhline(y=0, color='k')
# ax5.axvline(x=0, color='k')
# ax5.set(xlabel="t",ylabel="x")
# ax5.plot(t[np.where(t < t_drop/fps-tau[0])], r_pre[0,:], 'b')
# ax5.plot(t[np.where(t < t_drop/fps-tau[0])], np.squeeze(r_lin[0,np.where(t < t_drop/fps-tau[0])]), 'r')
# ax6.grid(True, which='both')
# ax6.axhline(y=0, color='k')
# ax6.axvline(x=0, color='k')
# ax6.set(xlabel="t",ylabel="y")
# ax6.plot(t[np.where(t < t_drop/fps-tau[0])], r_pre[1,:], 'b')
# ax6.plot(t[np.where(t < t_drop/fps-tau[0])], np.squeeze(r_lin[1,np.where(t < t_drop/fps-tau[0])]), 'r')

# fig8, (ax81,ax82) = plt.subplots(2,1)
# fig8.tight_layout()
# ax81.grid(True, which='both')
# ax81.axhline(y=0, color='k')
# ax81.axvline(x=0, color='k')
# ax81.set(xlabel="t",ylabel="x")
# ax81.plot(t[np.where(t >= t_drop/fps-tau[0])], r_post[0,:], 'b')
# ax81.plot(t[np.where(t >= t_drop/fps-tau[0])], np.squeeze(r_lin[0,np.where(t >= t_drop/fps-tau[0])]-r_lin[0,np.min(np.where(t >= t_drop/fps-tau[0]))]), 'r')
# ax82.grid(True, which='both')
# ax82.axhline(y=0, color='k')
# ax82.axvline(x=0, color='k')
# ax82.set(xlabel="t",ylabel="y")
# ax82.plot(t[np.where(t >= t_drop/fps-tau[0])], r_post[1,:], 'b')
# ax82.plot(t[np.where(t >= t_drop/fps-tau[0])], np.squeeze(r_lin[1,np.where(t >= t_drop/fps-tau[0])]-r_lin[1,np.min(np.where(t >= t_drop/fps-tau[0]))]), 'r')


# theta_s = np.arctan2(np.sin(theta_s), np.cos(theta_s))
# fig80, ax80 = plt.subplots()
# fig80.tight_layout()
# ax80.grid(True, which='both')
# ax80.axhline(y=0, color='k')
# ax80.axvline(x=0, color='k')
# ax80.set(xlabel="s",ylabel="$\\theta$")
# ax80.plot(s, theta_s*180/np.pi, 'b')
# # ax80.plot(t[np.where(t < t_drop/fps-tau[0])], kappa[np.where(t < t_drop/fps-tau[0])]*0.01/len_scale*180/np.pi, 'r')
# # ax80.plot(t[np.where(t >= t_drop/fps-tau[0])], kappa[np.where(t >= t_drop/fps-tau[0])]*0.01/len_scale*180/np.pi, 'b')

# fig80, ax80 = plt.subplots()
# fig80.tight_layout()
# ax80.grid(True, which='both')
# ax80.axhline(y=0, color='k')
# ax80.axvline(x=0, color='k')
# ax80.set(xlabel="s",ylabel="$\Delta \\theta$")
# ax80.plot(s, del_theta_s, 'c')
# ax80.plot(s, del_theta_s_ma, 'b')
# ax80.plot(s, del_theta_s_mstd, 'r')
# ax80.plot(s, -del_theta_s_mstd, 'r')
# # ax80.plot(s, del_theta_s*180/np.pi, 'r')
# # ax80.plot(s, kappa_reconst*0.01/len_scale*180/np.pi, 'b')
# # ax80.scatter(d_pellet_s, abs(kappa_s)*0.01/len_scale*180/np.pi, s=5)


# fig801, ax801 = plt.subplots()
# fig801.tight_layout()
# ax801.grid(True, which='both')
# ax801.axhline(y=0, color='k')
# ax801.axvline(x=0, color='k')
# ax801.set(xlabel="$\Delta \\theta$",ylabel="fraction")
# ax801.hist((omega_s*0.01/len_scale)*180/np.pi, bins=1000, weights=np.ones(len(omega_s)) / len(omega_s))
# # ax801.hist((kappa_s_scale*0.01/len_scale)*180/np.pi, bins=1000)
# # plt.xscale("symlog")
# # plt.yscale("symlog")
# plt.rc('axes', labelsize=15)

# theta_s = np.arctan2(np.sin(theta_s), np.cos(theta_s))
# theta_s_reconst = np.arctan2(np.sin(kappa_reconst_cumsum*0.01/len_scale), np.cos(kappa_reconst_cumsum*0.01/len_scale))
# fig81, ax81 = plt.subplots()
# fig81.tight_layout()
# ax81.grid(True, which='both')
# ax81.axhline(y=0, color='k')
# ax81.axvline(x=0, color='k')
# ax81.set(xlabel="s",ylabel="$\phi$(deg)")
# ax81.plot(s[np.where(s < s_drop)], theta_s[np.squeeze(np.where(s < s_drop))]*180/np.pi, 'r')
# ax81.plot(s[np.where(s >= s_drop)], theta_s[np.squeeze(np.where(s >= s_drop))]*180/np.pi, 'b')
# ax81.plot(s[np.where(s < s_drop)], theta_s_reconst[np.squeeze(np.where(s < s_drop))]*180/np.pi, 'k')
# ax81.plot(s[np.where(s >= s_drop)], theta_s_reconst[np.squeeze(np.where(s >= s_drop))]*180/np.pi, 'k')


# fig82, ax82 = plt.subplots()
# fig82.tight_layout()
# ax82.grid(True, which='both')
# ax82.axhline(y=0, color='k')
# ax82.axvline(x=0, color='k')
# ax82.set(xlabel="s",ylabel="$\phi$(rad)")
# ax82.plot(s[np.where(s < s_drop)], phi_pellet_s[np.squeeze(np.where(s < s_drop))], 'r')
# ax82.plot(s[np.where(s >= s_drop)], phi_pellet_s[np.squeeze(np.where(s >= s_drop))], 'b')

# fig7, (ax71,ax72) = plt.subplots(2,1)
# fig7.tight_layout()
# ax71.grid(True, which='both')
# ax71.axhline(y=0, color='k')
# ax71.axvline(x=0, color='k')
# ax71.set(xlabel="t",ylabel="$u_N$")
# ax71.plot(t[np.where(t < t_drop/fps-tau[0])], u_N[np.where(t < t_drop/fps-tau[0])], 'b')
# ax71.plot(t[np.where(t >= t_drop/fps-tau[0])], u_N[np.where(t >= t_drop/fps-tau[0])], 'r')
# ax72.grid(True, which='both')
# ax72.axhline(y=0, color='k')
# ax72.axvline(x=0, color='k')
# ax72.set(xlabel="t",ylabel="$u_T$")
# ax72.plot(t[np.where(t < t_drop/fps-tau[0])], u_T[np.where(t < t_drop/fps-tau[0])], 'b')
# ax72.plot(t[np.where(t >= t_drop/fps-tau[0])], u_T[np.where(t >= t_drop/fps-tau[0])], 'r')



# fig10, (ax101,ax102) = plt.subplots(2,1)
# fig10.tight_layout()
# ax101.grid(True, which='both')
# ax101.axhline(y=0, color='k')
# ax101.axvline(x=0, color='k')
# ax101.set(xlabel="t",ylabel="$nest_{rel_N}$")
# ax101.plot(t[np.where(t < t_drop/fps-tau[0])], nest_N[np.where(t < t_drop/fps-tau[0])], 'b')
# ax101.plot(t[np.where(t >= t_drop/fps-tau[0])], nest_N[np.where(t >= t_drop/fps-tau[0])], 'r')
# ax102.grid(True, which='both')
# ax102.axhline(y=0, color='k')
# ax102.axvline(x=0, color='k')
# ax102.set(xlabel="t",ylabel="$nest_{rel_T}$")
# ax102.plot(t[np.where(t < t_drop/fps-tau[0])], nest_T[np.where(t < t_drop/fps-tau[0])], 'b')
# ax102.plot(t[np.where(t >= t_drop/fps-tau[0])], nest_T[np.where(t >= t_drop/fps-tau[0])], 'r')




# fig20, (ax201,ax202) = plt.subplots(2,1)
# fig20.tight_layout()
# ax201.grid(True, which='both')
# ax201.axhline(y=0, color='k')
# ax201.axvline(x=0, color='k')
# ax201.set(xlabel="t",ylabel="$feeder_{rel_N}$")
# ax201.plot(t[np.where(t < t_drop/fps-tau[0])], feeder_N[np.where(t < t_drop/fps-tau[0])], 'b')
# ax201.plot(t[np.where(t >= t_drop/fps-tau[0])], feeder_N[np.where(t >= t_drop/fps-tau[0])], 'r')
# ax202.grid(True, which='both')
# ax202.axhline(y=0, color='k')
# ax202.axvline(x=0, color='k')
# ax202.set(xlabel="t",ylabel="$feeder_{rel_T}$")
# ax202.plot(t[np.where(t < t_drop/fps-tau[0])], feeder_T[np.where(t < t_drop/fps-tau[0])], 'b')
# ax202.plot(t[np.where(t >= t_drop/fps-tau[0])], feeder_T[np.where(t >= t_drop/fps-tau[0])], 'r')




# fig30, (ax301,ax302) = plt.subplots(2,1)
# fig30.tight_layout()
# ax301.grid(True, which='both')
# ax301.axhline(y=0, color='k')
# ax301.axvline(x=0, color='k')
# ax301.set(xlabel="t",ylabel="$pellet_{rel_N}$")
# ax301.plot(t[np.where(t < t_drop/fps-tau[0])], pellet_N[np.where(t < t_drop/fps-tau[0])], 'b')
# ax301.plot(t[np.where(t >= t_drop/fps-tau[0])], pellet_N[np.where(t >= t_drop/fps-tau[0])], 'r')
# ax302.grid(True, which='both')
# ax302.axhline(y=0, color='k')
# ax302.axvline(x=0, color='k')
# ax302.set(xlabel="t",ylabel="$pellet_{rel_T}$")
# ax302.plot(t[np.where(t < t_drop/fps-tau[0])], pellet_T[np.where(t < t_drop/fps-tau[0])], 'b')
# ax302.plot(t[np.where(t >= t_drop/fps-tau[0])], pellet_T[np.where(t >= t_drop/fps-tau[0])], 'r')


# fig8, ax8 = plt.subplots()
# ax8.grid(True, which='both')
# ax8.axhline(y=0, color='k')
# ax8.axvline(x=0, color='k')
# ax8.set(xlabel="x",ylabel="y")
# ax8.set_autoscale_on(False)
# ax8.axis([xmin,xmax,ymin,ymax])
# ax8.plot(r_pre[0,np.squeeze(np.where( (t>=t_change) & (t<t_drop/fps-tau[0]) ))], r_pre[1,np.squeeze(np.where( (t>=t_change) & (t<t_drop/fps-tau[0]) ))]  , zorder=1)
# ax8.imshow(np.rot90(rho_pre), cmap=plt.cm.gist_earth_r, extent=[xmin, xmax, ymin, ymax])
# ax8.set_xlim([xmin, xmax])
# ax8.set_ylim([ymin, ymax])



# # Average orientation correlation
delay_pre = np.array(range( -fl(len(tangent_corr_pre)/2), fl(len(tangent_corr_pre)/2)+1 ))*0.01
delay_pre_int = delay_pre[np.squeeze(np.where( delay_pre >=0 ))]
delay_pre_int = delay_pre_int[:int(delay_pre_int.shape[0]/2)+1]
tangent_corr_pre_int = tangent_corr_pre[np.squeeze(np.where( delay_pre >=0 ))]
tangent_corr_pre_int = tangent_corr_pre_int[:int(tangent_corr_pre_int.shape[0]/2)+1]

tangent_corr_pre_envelope_int = tangent_corr_pre_envelope[np.squeeze(np.where( delay_pre >=0 ))]
tangent_corr_pre_envelope_int = tangent_corr_pre_envelope_int[:int(tangent_corr_pre_envelope_int.shape[0]/2)+1]

tangent_corr_pre_int_fft = fft(tangent_corr_pre_int)
pre_fft_peaks = find_peaks(abs(tangent_corr_pre_int_fft[0:tangent_corr_pre_int.shape[0]//2]))


# fig_stat5, stat5 = plt.subplots()
# stat5.grid(True, which='both')
# stat5.axhline(y=0, color='k')
# stat5.axvline(x=0, color='k')
# # stat5.set(xlabel="$s$ (arc shift)",ylabel="corr($\hat{t}_{pre}$,$\hat{t}_{pre}$)")
# # stat5.plot(delay_pre_int, tangent_corr_pre_int, 'b')
# # stat5.plot(delay_pre_int, ifft(tangent_corr_pre_int_fft[0:10], delay_pre_int.shape[0]), 'b')
# stat5.set(xlabel="Normalized Frequency (Steplength/Distance)",ylabel="PSD($\hat{t}$,$\hat{t}$)")
# stat5.plot(fftfreq(tangent_corr_pre_int.shape[0], 1)[:tangent_corr_pre_int.shape[0]//2], np.abs(tangent_corr_pre_int_fft[0:tangent_corr_pre_int.shape[0]//2]), 'r')
# stat5.set_xscale('log')
# stat5.set_yscale('log')

delay_post = np.array(range( -fl(len(tangent_corr_post)/2), fl(len(tangent_corr_post)/2)+1 ))*0.01
delay_post_int = delay_post[np.squeeze(np.where( delay_post >=0 ))]
delay_post_int = delay_post_int[:int(delay_post_int.shape[0]/2)+1]
tangent_corr_post_int = tangent_corr_post[np.squeeze(np.where( delay_post >=0 ))]
tangent_corr_post_int = tangent_corr_post_int[:int(tangent_corr_post_int.shape[0]/2)+1]
tangent_corr_post_int_fft = fft(tangent_corr_post_int)

# fig_stat6, stat6 = plt.subplots()
# stat6.grid(True, which='both')
# stat6.axhline(y=0, color='k')
# stat6.axvline(x=0, color='k')
# # stat6.set(xlabel="$s$ (arc shift)",ylabel="corr($\hat{t}_{post}$,$\hat{t}_{post}$)")
# # stat6.plot(delay_post_int, tangent_corr_post_int, 'b')
# # stat6.plot(delay_post_int, ifft(tangent_corr_post_int_fft), 'b')
# stat6.set(xlabel="Normalized Frequency (Steplength/Distance)",ylabel="PSD($\hat{t}$,$\hat{t}$)")
# stat6.plot(fftfreq(tangent_corr_post_int.shape[0], 1)[:tangent_corr_post_int.shape[0]//2], np.abs(tangent_corr_post_int_fft[0:tangent_corr_post_int.shape[0]//2]), 'r')
# stat6.set_xscale('log')
# stat6.set_yscale('log')
#
# np.savetxt(input_dir + 'tangent_corr_pre' + '.csv', np.array([fftfreq(tangent_corr_pre_int.shape[0], 1)[:tangent_corr_pre_int.shape[0]//2], np.abs(tangent_corr_pre_int_fft[0:tangent_corr_pre_int.shape[0]//2])]), delimiter=',')
# np.savetxt(input_dir + 'tangent_corr_post' + '.csv', np.array([fftfreq(tangent_corr_post_int.shape[0], 1)[:tangent_corr_post_int.shape[0]//2], np.abs(tangent_corr_post_int_fft[0:tangent_corr_post_int.shape[0]//2])]), delimiter=',')


np.savetxt('data/cumulative/' + type + 'tangent_corr_pre/' + 'tangent_corr_pre_s_' + str(num) + '.csv', tangent_corr_pre_int, delimiter=',')
np.savetxt('data/cumulative/' + type + 'tangent_corr_post/' + 'tangent_corr_post_s_' + str(num) + '.csv', tangent_corr_post_int, delimiter=',')

plt.plot(tangent_corr_post_int)
plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')
plt.show()

sys.exit()

# Other
# fig75, ax751 = plt.subplots()
# # ax751.grid(True, which='both')
# ax751.set(xlabel="arclength",ylabel="scale")
# # is751 = ax751.imshow(np.abs(kappa_cwt), cmap='RdYlBu_r', extent=[0,s[-1],scale_min,scale_max], aspect='auto', norm=colors.LogNorm(vmin=np.maximum(np.abs(kappa_cwt).min(),1e-2), vmax=np.abs(kappa_cwt).max()))
# is751 = ax751.imshow(np.abs(tangent_corr_pre_cwt), cmap='RdYlBu_r', extent=[0,s[-1],scale_min,scale_max], aspect='auto', norm=colors.LogNorm(vmin=np.maximum(np.abs(tangent_corr_pre_cwt).min(),1e-2), vmax=np.abs(tangent_corr_pre_cwt).max()))
# # is751 = ax751.imshow(np.abs(kappa_cwt), cmap='RdYlBu_r', extent=[0,s[-1],scale_max,scale_min], aspect='auto', vmax=abs(kappa_cwt).max(), vmin=abs(kappa_cwt).min())
# fig75.colorbar(is751, ax=ax751)

# fig85, ax851 = plt.subplots()
# # ax851.grid(True, which='both')
# ax851.set(xlabel="arclength",ylabel="scale")
# is851 = ax851.imshow(np.angle(kappa_cwt,deg=True), cmap='RdYlBu_r', extent=[0,s[-1],scale_min,scale_max], aspect='auto', vmax=np.angle(kappa_cwt,deg=True).max(), vmin=np.angle(kappa_cwt,deg=True).min())
# fig85.colorbar(is851, ax=ax851)
# # plt.colorbar(cax=plt.axes([0,s[-1],scale_max,scale_min]))


# x = np.linspace(start=0, stop=s[-1], num=kappa_haar.shape[1])
# y = np.linspace(start=0, stop=haar_lvl-1, num=haar_lvl)
# X, Y = np.meshgrid(x, y)
# fig95, ax951 = plt.subplots()
# # ax751.grid(True, which='both')
# ax951.set(xlabel="arclength",ylabel="scale")
# ax951.set_yticks(np.arange(0,haar_lvl))
# pc951 = ax951.pcolormesh(X, Y, abs(kappa_haar), norm=colors.SymLogNorm(linthresh=1e0, linscale=1, vmin=abs(kappa_haar).min(), vmax=abs(kappa_haar).max()), cmap='RdYlBu_r')
# # pc951 = ax951.pcolormesh(X, Y, kappa_haar, norm=colors.SymLogNorm(linthresh=1e0, linscale=1, vmin=kappa_haar.min(), vmax=kappa_haar.max()), cmap='RdYlBu_r')
# fig95.colorbar(pc951, ax=ax951)
# # ax951.pcolormesh(X, Y, kappa_haar, norm=colors.LogNorm(vmin=np.maximum(kappa_haar.min(),1e-2), vmax=kappa_haar.max()), cmap='RdYlBu_r')
# # fig95.colorbar(plt.cm.ScalarMappable(norm=colors.LogNorm(vmin=np.maximum(kappa_haar.min(),1e-2), vmax=kappa_haar.max()), cmap='RdYlBu_r'), ax=ax951)



plt.show()

# t_change = np.array([t_change], dtype='float')
# np.savetxt(input_dir + 't_change' + '.csv', t_change, delimiter=',')


# np.savetxt('data/cumulative/' + type + 'all/' + 'rel_nest' + '_' + str(num) + '.csv', (nest[0], nest[1]), delimiter=',')
# np.savetxt('data/cumulative/' + type + 'all/' + 'rel_r_pre_mean' + '_' + str(num) + '.csv', (r_pre_mean[0], r_pre_mean[1]), delimiter=',')
# np.savetxt('data/cumulative/' + type + 'all/' + 'rel_r_post_mean' + '_' + str(num) + '.csv', (r_post_mean[0], r_post_mean[1]), delimiter=',')

# kappa_s = kappa_s/len_scale
# np.savetxt(input_dir + 'kappa_s' + '.csv', kappa_s[np.where(s >= s_drop)], delimiter=',')
