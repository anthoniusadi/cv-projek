import matplotlib.pyplot as plt

def kalibrasi():
    pass

data_x = [10,14,16,20,25,28,31,34]
# kertas
data_yy = [77,56,48,39,31,28,20,19]
# bbox
data_y = [71,54,43,33,24,22,18,13]

fig = plt.figure()
fig.set_figheight(15)
fig.set_figwidth(10)
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

ax1.set_title('pixel to cm bbox')
ax1.scatter(x=data_x, y=data_y,c='blue')
ax1.set_xlabel('jarak dalam cm')
ax1.set_ylabel('panjang dalam pixel')

ax2.set_title('pixel to cm kertas')
ax2.scatter(x=data_x, y=data_yy,c='blue')
ax2.set_xlabel('jarak dalam cm')
ax2.set_ylabel('panjang dalam pixel')

plt.savefig('plot_pixel_to_cm.png')
plt.show()


