import input_data
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib

# Helper function to display a zip code with predictions
def display_zip(images,labels):
	# Get each digit
	tmp1 = images[0]
	tmp1 = tmp1.reshape((28,28))

	tmp2 = images[1]
	tmp2 = tmp2.reshape((28,28))

	tmp3 = images[2]
	tmp3 = tmp3.reshape((28,28))

	tmp4 = images[3]
	tmp4 = tmp4.reshape((28,28))

	tmp5 = images[4]
	tmp5 = tmp5.reshape((28,28))

	# Add each digit to the figure
	fig = plt.figure()
	a = fig.add_subplot(1,5,1)
	plt.imshow(tmp1, cmap = cm.Greys)
	a.set_title(labels[0])

	a = fig.add_subplot(1,5,2)
	plt.imshow(tmp2, cmap = cm.Greys)
	a.set_title(labels[1])

	a = fig.add_subplot(1,5,3)
	plt.imshow(tmp3, cmap = cm.Greys)
	a.set_title(labels[2])

	a = fig.add_subplot(1,5,4)
	plt.imshow(tmp4, cmap = cm.Greys)
	a.set_title(labels[3])

	a = fig.add_subplot(1,5,5)
	plt.imshow(tmp5, cmap = cm.Greys)
	a.set_title(labels[4])

	# Show the figure
	plt.show()