### 一、直方图匹配法（Histogram matching）

[链接]: https://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_histogram_matching.html

​	（1）从测试集中挑选一些具有代表性的图片集合D，这些图片的直方图之间差异比较大（可以选择全部的测试集）

​	（2）对于D中每一张图片d，将d作为reference，使用直方图匹配的方法匹配到训练集的图片上，这样训练集中的图片image经过处理变成和d具有相似直方图特征的图片matched，代码如下：

~~~python
matched = match_histograms(image, reference, multichannel=True)
~~~

​	（3）在目前最优的模型的基础上，将处理过的图片**加入到原训练集**中一起训练。



### 二、YUV亮度特征匹配

[链接]: https://docs.opencv.org/3.4/d8/d01/group__imgproc__color__conversions.html

​	(1)	向将训练集和测试的RGB图像都转为YUV图像

~~~python
yuv=cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
~~~

​	(2)	依照上面"直方图匹配"的方法对YUV数据集做类似的直方图匹配

​	(3)	做完后将生成的YUV图像再转回RGB

~~~python
RGB=cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
~~~

​	(4)	在目前最优的模型的基础上，再将这些图片**加入到原数据集**中继续训练