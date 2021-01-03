def tensor_to_image(tensor):
    """
    Converts a tensor to numpy and unormalize it
    :param tensor: Tensor in the format (3, W, H)
    :return: cv2 image in the format (W, H, 3) with values between 0 and 255
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    for i in range(3):
        tensor[i, :, :] = tensor[i, :, :] * std[i] + mean[i]
    img_np = tensor.permute(1, 2, 0).cpu().numpy()
    img_np = img_np * 255
    img_np = img_np[:, :, ::-1]
    return img_np.astype("uint8")
