# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 17:09:25 2019

@author: epinsky
"""


import numpy as np
import os
import matplotlib.pyplot as plt


input_dir = r'C:\Users\epinsky\bu\python\data_science_with_Python\plots'

X = np.array([1,2,3,4])
Y = np.array([5,2,9,10])


file_name = os.path.join(input_dir, 'labaled_weeks.pdf')
fig = plt.figure()
ax = plt.gca()
#ax.hold(True)


ax.set_xlim(0.5 * min(X), 1.5*max(X))
ax.set_ylim(0.5*min(Y), 1.5*max(Y))


ax.set_xlabel('X')
ax.set_ylabel('Y')

text_font = {'fontname':'Arial', 'size':'12', 
             'color':'black', 'weight':'normal',
              'verticalalignment':'top'} # Bottom vertical alignment for more space
# ax.view_init(0, 0)

size=500
#xx, yy, zz = np.meshgrid(range(4,8, 1), range(90,200,10), range(5, 12 ,1 ))
#plt3d.plot_surface(xx, yy, zz)

for i in range(len(X)):
    color = 'red'     # compute via function
    size = 300        # compute via function
    ax.scatter(X[i], Y[i], color=color, s = 300)


fig.show()
fig.savefig(file_name)

