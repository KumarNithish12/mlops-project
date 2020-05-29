import os

grepacc = os.popen("cat /root/Desktop/code/accuracy.txt")
grepacc1 = grepacc.read()
grepacc2 = grepacc1.rstrip()
grepacc3 = float(grepacc2)

if grepacc3<93:
	x = os.popen("cat /root/Desktop/code/mltrain.py | grep model.add | wc -l")
	x1 = x.read()
	x2 = x1.rstrip()
	x3 = int(x2)
	print("previously, No. of layers: "  , x3)

	if x3 == 2:
		y = 'model.add(Dense(units=32 , activation=\"relu\"))'
	elif x3 == 3:
		y = 'model.add(Dense(units=32 , activation=\"relu\"))'
	elif x3 == 4:
		y = 'model.add(Dense(units=16 , activation=\"relu\"))'
	elif x3 == 5:
		y = 'model.add(Dense(units=8 , activation=\"relu\"))'
	else:
		print("Dense layers are Enough for training")
		exit()

	os.system("sed -i '/softmax/ i {}' /root/Desktop/code/mltrain.py".format(y))
	print("... Lines are added ...")

else:
	print(".. Accuracy is above 90% ..")
