import numpy as np


def expand(x):
    return x * 2.0 - 1.0


def shrink(x):
    return (x + 1.0) / 2.0


def augment(x, k, inverse=False):
    assert k<8
    if not inverse:
        if k//4==1:
            x = x.transpose(-1, -2)
    if k//2==1:
        x = x.flip(-1)
    elif k//2==2:
        x = x.flip(-2)
    elif k//2==3:
        x = x.flip(-1, -2)
    if inverse:
        if k//4==1:
            x = x.transpose(-1, -2)
    return x


def extract_patches(image, patch_size, stride=None):
    image_height, image_width = image.shape[:2]
    if stride is None: stride = patch_size
    assert image.ndim==3
    assert patch_size>0 and stride>0
    patches = []
    for offset_height in range(0, image_height, stride):
        for offset_width in range(0, image_width, stride):
            if offset_height+patch_size>image_height or offset_width+patch_size>image_width: continue
            patches.append(image[offset_height:offset_height+patch_size, offset_width:offset_width+patch_size, :])
    if not (image_height-patch_size)%stride==0:
        for offset_width in range(0, image_width, stride):
            if offset_width+patch_size>image_width: continue
            patches.append(image[-patch_size:, offset_width:offset_width+patch_size, :])
    if not (image_width-patch_size)%stride==0:
        for offset_height in range(0, image_height, stride):
            if offset_height+patch_size>image_height: continue
            patches.append(image[offset_height:offset_height+patch_size:, -patch_size:, :])
    if (not (image_height-patch_size)%stride==0) and (not (image_width-patch_size)%stride==0):
        patches.append(image[-patch_size:, -patch_size:, :])
    return patches


def reconstruct_patches(patches, image_size, stride=None, weighting=None):
    if isinstance(patches, list): patches = np.stack(patches)
    num_patches = patches.shape[0]
    patch_size = patches.shape[1]
    image_height, image_width = image_size[:2]
    if stride is None: stride = patch_size
    assert patch_size>0 and stride>0
    assert weighting in [None, 'lin', 'cos']
    reconstructed_image = np.zeros([image_height, image_width, 3], dtype=patches.dtype)

    if weighting is not None:
        overlap = patch_size - stride
        assert patch_size >= overlap*2
        if weighting == 'lin':
            weight_vector = np.concatenate([np.linspace(0, 1, overlap, dtype=patches.dtype), np.ones(patch_size-overlap*2, dtype=patches.dtype), np.linspace(1, 0, overlap, dtype=patches.dtype)])
        elif weighting == 'cos':
            weight_vector = np.concatenate([(np.cos(np.linspace(-np.pi, 0, overlap))+1)/2.0, np.ones(patch_size-overlap*2, dtype=patches.dtype), (np.cos(np.linspace(0, np.pi, overlap))+1)/2.0])

        left_weight_vector, right_weight_vector = np.split(weight_vector, 2)
        left_weight = np.stack([np.concatenate([np.stack([left_weight_vector]*patch_size, axis=0), np.ones([patch_size, patch_size//2])], axis=1)]*3, axis=-1)
        right_weight = np.stack([np.concatenate([np.ones([patch_size, patch_size//2]), np.stack([right_weight_vector]*patch_size, axis=0)], axis=1)]*3, axis=-1)
        upper_weight = np.stack([np.concatenate([np.stack([left_weight_vector]*patch_size, axis=1), np.ones([patch_size//2, patch_size])], axis=0)]*3, axis=-1)
        lower_weight = np.stack([np.concatenate([np.ones([patch_size//2, patch_size]), np.stack([right_weight_vector]*patch_size, axis=1)], axis=0)]*3, axis=-1)

    current_patch_idx = 0
    for offset_height in range(0, image_height, stride):
        for offset_width in range(0, image_width, stride):
            if offset_height+patch_size>image_height or offset_width+patch_size>image_width: continue
            patch = patches[current_patch_idx]
            if weighting is not None:
                patch *= lower_weight
                patch *= right_weight
                if not offset_height==0:
                    patch *= upper_weight
                if not offset_width==0:
                    patch *= left_weight
                reconstructed_image[offset_height:offset_height+patch_size, offset_width:offset_width+patch_size] += patch
            else:
                reconstructed_image[offset_height:offset_height+patch_size, offset_width:offset_width+patch_size] = patch
            current_patch_idx += 1

    if not (image_height-patch_size)%stride==0:
        height_modulo = stride-(image_height-patch_size)%stride
        for offset_width in range(0, image_width, stride):
            if offset_width+patch_size>image_width: continue
            patch = patches[current_patch_idx][height_modulo:, :, :]
            patch *= upper_weight[:-height_modulo, :, :]
            patch *= right_weight[:-height_modulo, :, :]
            if not offset_width==0:
                patch *= left_weight[:-height_modulo, :, :]
            reconstructed_image[-(patch_size-height_modulo):, offset_width:offset_width+patch_size, :] += patch
            current_patch_idx += 1

    if not (image_width-patch_size)%stride==0:
        width_modulo = stride-(image_width-patch_size)%stride
        for offset_height in range(0, image_height, stride):
            if offset_height+patch_size>image_height: continue
            patch = patches[current_patch_idx][:, width_modulo:, :]
            patch *= lower_weight[:, :-width_modulo, :]
            patch *= left_weight[:, :-width_modulo, :]
            if not offset_height==0:
                patch *= upper_weight[:,:-width_modulo, :]
            reconstructed_image[offset_height:offset_height+patch_size, -(patch_size-width_modulo):, :] += patch
            current_patch_idx += 1

    if (not (image_width-patch_size)%stride==0) and (not (image_height-patch_size)%stride==0):
        patch = patches[current_patch_idx][height_modulo:, width_modulo:, :]
        patch *= upper_weight[:-height_modulo, :-width_modulo, :]
        patch *= left_weight[:-height_modulo, :-width_modulo, :]
        reconstructed_image[-(patch_size-height_modulo):, -(patch_size-width_modulo):, :] += patch
        current_patch_idx += 1

    assert num_patches == current_patch_idx
    return reconstructed_image
