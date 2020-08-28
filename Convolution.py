import numpy as np

def convolution_matrix(image, kernel, stride, h_out, w_out):
    out_matrix_temp = np.zeros((h_out, w_out))
    ker_size = np.shape(kernel)[0]
    for y_beg, num_y in zip(range(0, image.shape[0] - ker_size, stride), range(0, h_out, 1)):
        for x_beg, num_x in zip(range(0, image.shape[1] - ker_size, stride), range(0, w_out, 1)):
            split_matrix = image[x_beg:x_beg + ker_size, y_beg:y_beg + ker_size]
            corr = 0
            for x in range(0, ker_size, 1):
                for y in range(0, ker_size, 1):
                    corr += int(split_matrix[x][y] * kernel[x][y])
            out_matrix_temp[num_y][num_x] = corr
    return out_matrix_temp

def image_with_padding(image, padding):
    h_padding = np.zeros([image.shape[0], padding])
    new_image = np.hstack((h_padding, image, h_padding))
    w_padding = np.zeros([padding, image.shape[1] + 2 * padding])
    new_image = np.vstack((w_padding, new_image, w_padding))
    return new_image

def сonvolution(image, ker_size=4, stride=4, padding=2, channels_out=1):
    # hyperparameters
    batch = image.shape[0]
    channels_in = image.shape[1]
    y=image.shape[2]
    x=image.shape[3]
    # new_size
    h_out = (y + 2 * padding - ker_size) // stride + 1
    w_out = (x + 2 * padding - ker_size) // stride + 1
    # weights generation
    kernel = np.random.random((channels_out, ker_size, ker_size))
    bias = np.random.random((channels_out))
    out_matrix = np.zeros((batch, channels_out, h_out, w_out))
    # main loop
    for num_batch in range(0, batch):
        for num_channel_out in range(0, channels_out):
            for num_channel_in in range(0, channels_in):
                new_kernel = kernel[num_channel_in][:][:]
                new_image = image_with_padding(image[num_batch][num_channel_in][:][:], padding)
                new_matrix = convolution_matrix(new_image, new_kernel, stride, h_out, w_out)
                out_matrix[num_batch] += new_matrix
            out_matrix[num_batch] += bias
    return out_matrix