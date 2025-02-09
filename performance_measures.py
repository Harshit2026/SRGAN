import os
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.metrics import mean_squared_error, roc_curve, auc
from keras.applications import VGG19
from keras.models import Model

# 1️⃣ Load the Trained Generator Model
generator = load_model("/kaggle/input/srgan/keras/default/1/gen_e_25.h5")  # Update path if needed
discriminator = load_model("/kaggle/input/srgan/keras/default/1/disc_e_25.h5")  # Load Discriminator

# 2️⃣ Load Test Images from Kaggle Dataset
lr_dir = "/kaggle/working/lr_images"  # Update directory
hr_dir = "/kaggle/working/hr_images"

# Load 10 sample images
lr_filenames = sorted(os.listdir(lr_dir))[:10]
hr_filenames = sorted(os.listdir(hr_dir))[:10]

lr_test = np.array([cv2.cvtColor(cv2.imread(os.path.join(lr_dir, img)), cv2.COLOR_BGR2RGB) for img in lr_filenames])
hr_test = np.array([cv2.cvtColor(cv2.imread(os.path.join(hr_dir, img)), cv2.COLOR_BGR2RGB) for img in hr_filenames])

# Normalize images (Scale pixel values to [0,1])
lr_test, hr_test = lr_test / 255.0, hr_test / 255.0

# 3️⃣ Generate Super-Resolved Images using Generator
pred_images = generator.predict(lr_test)
pred_images = np.clip(pred_images, 0, 1)  # Ensure valid pixel range

# 4️⃣ Load VGG19 Model (Pretrained on ImageNet) and Compute Perceptual Loss
vgg = VGG19(weights="imagenet", include_top=False, input_shape=(hr_test.shape[1], hr_test.shape[2], 3))
feature_extractor = Model(inputs=vgg.input, outputs=vgg.layers[10].output)  # Extract mid-level features

perceptual_loss_scores = [
    np.mean(np.square(feature_extractor.predict(hr_test[i:i+1]) - feature_extractor.predict(pred_images[i:i+1])))
    for i in range(len(pred_images))
]

# 5️⃣ Compute Performance Metrics
psnr_scores = [psnr(hr_test[i], pred_images[i], data_range=1.0) for i in range(len(pred_images))]
ssim_scores = [ssim(hr_test[i], pred_images[i], channel_axis=2, data_range=1.0) for i in range(len(pred_images))]
mse_scores = [mean_squared_error(hr_test[i].flatten(), pred_images[i].flatten()) for i in range(len(pred_images))]

# 6️⃣ Create Performance Table
df = pd.DataFrame({
    "Filename": lr_filenames,
    "PSNR": psnr_scores,
    "SSIM": ssim_scores,
    "MSE": mse_scores,
    "Perceptual Loss": perceptual_loss_scores
})

print("\nPerformance Metrics Table:")
print(df)

# 7️⃣ Plot Pixel Difference Histogram
plt.figure(figsize=(8, 5))
for i in range(len(pred_images)):
    plt.hist(np.abs(hr_test[i] - pred_images[i]).ravel(), bins=50, alpha=0.5, label=f"Image {i+1}")

plt.title("Pixel Difference Histogram")
plt.xlabel("Pixel Difference")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# 8️⃣ Plot Performance Metrics Distribution
plt.figure(figsize=(10, 5))
plt.hist(psnr_scores, bins=20, alpha=0.7, label="PSNR")
plt.hist(ssim_scores, bins=20, alpha=0.7, label="SSIM")
plt.hist(mse_scores, bins=20, alpha=0.7, label="MSE")

# Add labels and title
plt.xlabel("Metric Value")  # X-axis: Metric values (PSNR, SSIM, MSE)
plt.ylabel("Frequency")  # Y-axis: How often each value appears
plt.legend()
plt.title("Performance Metrics Distribution")
plt.show()


# 9️⃣ Display Example Images (Low-Res, Super-Res, High-Res)
fig, axes = plt.subplots(3, 5, figsize=(15, 10))
for i in range(5):
    axes[0, i].imshow(lr_test[i])
    axes[0, i].set_title("Low-Res")
    axes[1, i].imshow(pred_images[i])
    axes[1, i].set_title(f"Super-Res\nPSNR: {psnr_scores[i]:.2f}, SSIM: {ssim_scores[i]:.2f}")
    axes[2, i].imshow(hr_test[i])
    axes[2, i].set_title("High-Res (GT)")
for ax in axes.flatten():
    ax.axis("off")
plt.show()

# 🔟 Print Average Scores
print(f"Avg PSNR: {np.mean(psnr_scores):.2f}")
print(f"Avg SSIM: {np.mean(ssim_scores):.2f}")
print(f"Avg MSE: {np.mean(mse_scores):.5f}")
print(f"Avg Perceptual Loss: {np.mean(perceptual_loss_scores):.5f}")

# 1️⃣1️⃣ Compute ROC Curve for Discriminator
# Prepare Labels (1 for Real HR Images, 0 for Generated SR Images)
real_labels = np.ones((len(hr_test), 1))  # Ground Truth HR Images
fake_labels = np.zeros((len(pred_images), 1))  # Generated SR Images

# Get Discriminator Scores
real_scores = discriminator.predict(hr_test)
fake_scores = discriminator.predict(pred_images)

# Combine Labels and Scores
y_true = np.vstack((real_labels, fake_labels)).flatten()
y_scores = np.vstack((real_scores, fake_scores)).flatten()

# Compute ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")  # Random classifier line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Discriminator")
plt.legend(loc="lower right")
plt.show()
