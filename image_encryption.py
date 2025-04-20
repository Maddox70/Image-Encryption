import cv2
import numpy as np
import matplotlib.pyplot as plt

def arnold_cat_map_matrix(image, iterations):
    h, w, c = image.shape
    scrambled = image.copy()
    matrix = np.array([[1, 1], [1, 2]])
    for _ in range(iterations):
        temp = np.zeros_like(image)
        for i in range(h):
            for j in range(w):
                new_coords = np.dot(matrix, np.array([i, j])) % np.array([h, w])
                temp[new_coords[0], new_coords[1]] = scrambled[i, j]
        scrambled = temp.copy()
    return scrambled

def inverse_arnold_cat_map_matrix(image, iterations):
    h, w, c = image.shape
    descrambled = image.copy()
    matrix = np.array([[2, -1], [-1, 1]])  # Inverse matrix
    for _ in range(iterations):
        temp = np.zeros_like(image)
        for i in range(h):
            for j in range(w):
                new_coords = np.dot(matrix, np.array([i, j])) % np.array([h, w])
                temp[new_coords[0], new_coords[1]] = descrambled[i, j]
        descrambled = temp.copy()
    return descrambled

def logistic_map(size, key):
    r = 3.99
    seq = np.zeros(size)
    seq[0] = key
    for i in range(1, size):
        seq[i] = r * seq[i - 1] * (1 - seq[i - 1])
    return (seq * 256).astype(np.uint8)

def sine_map(size, key):
    seq = np.zeros(size)
    seq[0] = key
    for i in range(1, size):
        seq[i] = np.abs(np.sin(np.pi * seq[i - 1]))
    return (seq * 256).astype(np.uint8)

def encrypt_image(image, key):
    iterations = 2  # Reduced iterations
    scrambled = arnold_cat_map_matrix(image, iterations)
    h, w, c = scrambled.shape
    chaotic_seq = logistic_map(h * w * c, key) + sine_map(h * w * c, key / 2)
    chaotic_seq = chaotic_seq.reshape(h, w, c)
    encrypted = np.bitwise_xor(scrambled, chaotic_seq)
    return encrypted

def decrypt_image(encrypted_image, key):
    h, w, c = encrypted_image.shape
    chaotic_seq = logistic_map(h * w * c, key) + sine_map(h * w * c, key / 2)
    chaotic_seq = chaotic_seq.reshape(h, w, c)
    descrambled = np.bitwise_xor(encrypted_image, chaotic_seq)
    iterations = 2  # Must match encryption iterations
    original = inverse_arnold_cat_map_matrix(descrambled, iterations)
    return original

if __name__ == "__main__":
    image = cv2.imread("input.bmp")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize to 256x256 (or another power of 2)
    image = cv2.resize(image, (256, 256))

    key = 0.67
    encrypted_image = encrypt_image(image, key)
    plt.imsave("encrypted.jpg", encrypted_image)
    decrypted_image = decrypt_image(encrypted_image, key)
    plt.imsave("decrypted.jpg", decrypted_image)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(image)
    axs[0].set_title("Original Image")
    axs[1].imshow(encrypted_image)
    axs[1].set_title("Encrypted Image")
    axs[2].imshow(decrypted_image)
    axs[2].set_title("Decrypted Image")
    for ax in axs:
        ax.axis("off")
    plt.show()