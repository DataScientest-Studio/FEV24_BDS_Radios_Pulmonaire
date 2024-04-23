'''
countplot_LABEL.py

Ce script permet d'afficher un countplot du nombre d'images par LABEL.
'''

ax = sns.countplot(x = 'LABEL',
                   data = df_images,
                   order = df_images['LABEL'].value_counts().index,
                   palette = ['#A1C9F4', '#8DE5A1', '#FFB482', '#D0BBFF'],
                   edgecolor = 'gray')
plt.title("Distribution du nombre d'images par LABEL")
plt.ylabel("Nombre d'images")
plt.xlabel("Label")

for p in ax.patches:
    ax.text(p.get_x() + p.get_width() / 2., p.get_height(), '%d' % int(p.get_height()),
            ha = 'center', va = 'bottom')

plt.show()