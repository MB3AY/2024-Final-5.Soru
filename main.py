import numpy as np

# Histogram verilerini oluşturun
histogram_data = {
    100: 12, 101: 18, 102: 32, 103: 48, 104: 52, 105: 65, 106: 55, 107: 42,
    108: 32, 109: 16, 110: 10, 140: 5, 141: 18, 142: 25, 143: 32, 144: 40,
    145: 65, 146: 43, 147: 32, 148: 20, 149: 10, 150: 4
}

# Yoğunluk ve piksel sayılarını ayır
intensity_values = np.array(list(histogram_data.keys()))
pixel_counts = np.array(list(histogram_data.values()))

# Toplam piksel sayısı
total_pixels = np.sum(pixel_counts)

# Yoğunluk değerlerinin toplamını hesaplayın
sum_intensity = np.sum(intensity_values * pixel_counts)

# Otsu eşikleme için değişkenleri başlat
current_min_variance = np.inf
optimal_threshold = 0

# Sınıf içi varyansı hesaplamak için döngü başlat
for t in range(1, len(intensity_values)):
    # Background (G1)
    weight_b = np.sum(pixel_counts[:t]) / total_pixels
    mean_b = np.sum(intensity_values[:t] * pixel_counts[:t]) / np.sum(pixel_counts[:t]) if np.sum(pixel_counts[:t]) != 0 else 0
    variance_b = np.sum(((intensity_values[:t] - mean_b) ** 2) * pixel_counts[:t]) / np.sum(pixel_counts[:t]) if np.sum(pixel_counts[:t]) != 0 else 0

    # Foreground (G2)
    weight_f = np.sum(pixel_counts[t:]) / total_pixels
    mean_f = np.sum(intensity_values[t:] * pixel_counts[t:]) / np.sum(pixel_counts[t:]) if np.sum(pixel_counts[t:]) != 0 else 0
    variance_f = np.sum(((intensity_values[t:] - mean_f) ** 2) * pixel_counts[t:]) / np.sum(pixel_counts[t:]) if np.sum(pixel_counts[t:]) != 0 else 0

    # Sınıf içi varyansı hesapla
    within_class_variance = weight_b * variance_b + weight_f * variance_f

    # Minimum varyans ve optimal eşik değerini güncelle
    if within_class_variance < current_min_variance:
        current_min_variance = within_class_variance
        optimal_threshold = intensity_values[t]

# Optimum eşik değerini yazdır
print(f"Optimum Eşik Değeri: {optimal_threshold}")
