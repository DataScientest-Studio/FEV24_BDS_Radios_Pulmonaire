'''
calculate_mask_ratio.py

Ce script permet de calculer le ratio de pixels différent de 0 pour évaluer le pourcentage de surface utile d'un masque.
'''

# Calcul des ratios de surface utile

RATIO = []

for index, mask_path in enumerate(df_masks['PATH']):
    img_msk = Image.open(mask_path)
    arr_msk = np.array(img_msk)
    msk_size = arr_msk.shape

    # calcul du ratio surface utile à partir du mask
    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask_arr = np.array(mask_img)
    mask_ratio = np.round(np.count_nonzero(mask_arr)/(299*299),4)*100
    RATIO.append(mask_ratio)

df_masks['RATIO'] = pd.Series(RATIO)