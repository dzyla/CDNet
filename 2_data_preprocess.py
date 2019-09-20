import numpy as np

def split_sequence(sequence1, sequence2, y_data, n_steps):
    X, y = list(), list()
    for i in range(len(sequence1)):
        #print(i)

        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence1) - 1:
            break
        # gather input and output parts of the pattern
        seq_x = sequence1[i:end_ix]
        seq_x2 = sequence2[i:end_ix]

        X.append([seq_x, seq_x2])
        y.append(np.split(y_data, 8))
    return np.array(X), np.array(y)



x_data = np.load('CDnet_xdata.npy', allow_pickle=True)
y_data = np.load('CDnet_ydata.npy', allow_pickle=True)

single = x_data[1]

final_array = []
y_array = []
first_iter = True


for n, single in enumerate(x_data):

    x1, labels1 = split_sequence(single[0],single[1], y_data[n], 5)
    ltsm_data = x1



    if first_iter:
        final_array = np.expand_dims(ltsm_data, axis=0)
        #print(final_array.shape)
        first_iter = False

        y_array = np.expand_dims(labels1, axis=0)

    else:
        ltsm_data = np.expand_dims(ltsm_data, axis=0)
        final_array = np.concatenate((final_array, ltsm_data), axis=0)


        labels1 = np.expand_dims(labels1, axis=0)
        y_array = np.concatenate((y_array, labels1), axis=0)



    print(final_array.shape)


final_array = np.reshape(final_array, (-1, 2, 5))
y_array = np.reshape(y_array, (-1, 8))


print(final_array.shape)
print(y_array.shape)


np.save('CD_data_ltsm.npy', final_array)
np.save('CD_data_ltsm_labels.npy', y_array)
