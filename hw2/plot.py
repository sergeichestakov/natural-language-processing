import matplotlib.pyplot as plt

word_error = [0.2816, 0.1082, 0.0814, 0.065, 0.057, 0.0540]
sentence_error = [0.9482, 0.8, 0.7494, 0.7035, 0.6676, 0.6558]

num_lines = [1000, 5000, 10000, 20000, 30000, 39832]

plt.plot(num_lines, word_error)
plt.plot(num_lines, sentence_error)

plt.xlabel('Number of lines from the corpus')
plt.ylabel('Percent error')

plt.title('Learning Curve')

plt.show()
