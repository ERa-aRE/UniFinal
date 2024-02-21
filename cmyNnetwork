import numpy
import scipy.special 
import matplotlib.pyplot

import scipy.misc
%matplotlib inline
#تعریف کلاس شبکه عصبی
class myNnetwork():
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        #مقدار دهی اولیه
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate
        self.wih = (numpy.random.rand(self.hnodes, self.inodes) - 0.5)
        self.who = (numpy.random.rand(self.onodes, self.hnodes) - 0.5)
        self.activation_function = lambda x: scipy.special.expit(x)
    def train(self,inputs_list, targets_list):
        #آموزش شبکه
        # تبدیل لیست ورودی به یک آرایه دو بعدی
        inputs = numpy.array(inputs_list, ndmin=2).T
        # تبدیل لیست هدف به یک آرایه دو بعدی
        targets = numpy.array(targets_list, ndmin=2).T
        # مشابه متد پرسش
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        #
        # محاسبه خطا
        output_errors = targets - final_outputs
        #خطای پس انتشار برای لایه ی پنهان 
        hidden_errors = numpy.dot(self.who.T, output_errors)
        # بروز رسانی وزن ها بین لایه ی پنهان و لایه ی خروجی 
        self.who += self.lr * numpy.dot((output_errors * final_outputs *(1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        # بروز رسانی وزن ها بین لایه ی ورودی و لایه ی پنهان
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs *(1.0 - hidden_outputs)), numpy.transpose(inputs))
        pass
    def query(self, inputs_list):
        #  تبدیل لیست ورودی به یک آرایه دو بعدی
        inputs = numpy.array(inputs_list, ndmin=2).T
        # محاسبه سیگنال های ورودی به داخل لایه نهان
        hidden_inputs = numpy.dot(self.wih, inputs)
        # محاسبه سیگنال های خروجی از لایه پنهان
        hidden_outputs = self.activation_function(hidden_inputs)
        # محاسبه سیگنال های ورودی به داخل لایه ی خروجی
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # محاسبه سیگنال های خروجی از لایه خروجی
        final_outputs = self.activation_function(final_inputs)
        return final_outputs
        pass
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.2
# ساخت یک شی از کلاس شبکه عصبی
xn = myNnetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)  
training_data_file =open('Downloads\mnist_train.csv','r')
training_data_list =training_data_file.readlines()
training_data_file.close()
for record in training_data_list:
    all_values = record.split(',')
    # اسکیل داده ها
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    targets = numpy.zeros(output_nodes) + 0.01
    targets[int(all_values[0])] = 0.99
    xn.train(inputs, targets)
    pass
