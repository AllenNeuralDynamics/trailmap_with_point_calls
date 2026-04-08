import numpy as np

def segment_brain_array(input_vol, model):
    # input_vol: numpy array of shape (z, y, x)
    # model: segmentation model
    # Returns: segmented volume of same shape

    input_dim = 64
    output_dim = 32

    # Pad input if needed
    dim_offset = (input_dim - output_dim) // 2
    temp_section = np.pad(input_vol, ((dim_offset, dim_offset),
                                      (dim_offset, dim_offset),
                                      (dim_offset, dim_offset)), 'edge')

    seg = np.zeros(temp_section.shape, dtype='float32')
    coords = []
    for x in range(0, temp_section.shape[1] - input_dim, output_dim):
        for y in range(0, temp_section.shape[2] - input_dim, output_dim):
            coords.append((0, x, y))
    for x in range(0, temp_section.shape[1] - input_dim, output_dim):
        coords.append((0, x, temp_section.shape[2]-input_dim))
    for y in range(0, temp_section.shape[2] - input_dim, output_dim):
        coords.append((0, temp_section.shape[1]-input_dim, y))
    coords.append((0, temp_section.shape[1]-input_dim, temp_section.shape[2]-input_dim))

    batch_size = 1
    threshold = 0.01
    batch_crops = np.zeros((batch_size, input_dim, input_dim, input_dim))
    batch_coords = np.zeros((batch_size, 3), dtype="int")
    i = 0

    while i < len(coords):
        batch_count = 0
        while i < len(coords) and batch_count < batch_size:
            (z, x, y) = coords[i]
            test_crop = temp_section[z:z + input_dim, x:x + input_dim, y:y + input_dim]
            if np.max(test_crop) > threshold:
                batch_coords[batch_count] = (z, x, y)
                batch_crops[batch_count] = test_crop
                batch_count += 1
            i += 1
        batch_input = np.reshape(batch_crops, batch_crops.shape + (1,))
        output = np.squeeze(model.predict(batch_input)[:, :, :, :, [0]])
        for j in range(batch_count):
            (z, x, y) = batch_coords[j] + dim_offset
            seg[z:z + output_dim, x:x + output_dim, y:y + output_dim] = output[j]

    cropped_seg = seg[dim_offset:dim_offset + input_vol.shape[0],
                      dim_offset:dim_offset + input_vol.shape[1],
                      dim_offset:dim_offset + input_vol.shape[2]]
    return cropped_seg