'''
show_img_samples.py

Ce script permet d'afficher aléatoirement 5 images de chaque classe du jeu de données.
'''

def sample_paths(group):
    return group.sample(n = 5)

# Sélectionner 5 chemins aléatoires pour chaque label
sampled_paths = df_images.groupby('LABEL')['PATH'].apply(sample_paths).reset_index(drop = True)

# Afficher 5 images aléatoires pour chaque label
for label, group in df_images.groupby('LABEL'):
    print(f"Label: {label}")
    sampled_paths = group['PATH'].sample(min(len(group), 5))
    fig, axs = plt.subplots(1, len(sampled_paths), figsize = (20, 20))
    for ax, path in zip(axs, sampled_paths):
        img = mpimg.imread(path)
        ax.imshow(img, cmap = "gray")
        ax.axis('off')
    plt.show()