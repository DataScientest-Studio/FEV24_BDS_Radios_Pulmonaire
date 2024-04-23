'''
pieplot_SOURCE.py

Ce script permet d'afficher un pieplot montrant la proportion de chaque source dans le jeu de données.
'''

source_counts = df_images['SOURCE'].value_counts()

sns.set_style("whitegrid")
sns.set_palette("Greys")
plt.figure(figsize = (8, 8))
plt.pie(source_counts,
        labels = source_counts.index,
        autopct = '%1.1f%%',
        wedgeprops = {'edgecolor': 'black'},
        textprops = {'weight': 'bold'})
plt.show()

