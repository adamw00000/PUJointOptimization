import numpy as np
import seaborn as sns

palette = sns.color_palette('colorblind')

is_metric_increasing = {
    'Błąd aproksymacji (AE) prawdopodobieństwa a posteriori': True,
    r'Błąd estymacji częstości etykietowania': True,
    r'Błąd estymacji prawdopodobieństwa a priori': True,
    'AUC': False,
    'Czas wykonania': True,
    'Iteracje metody': True,
    'Ewaluacje funkcji w trakcie optymalizacji': True,
}

metric_ylim = {
    'Błąd aproksymacji (AE) prawdopodobieństwa a posteriori': (0, 0.5),
    r'Błąd estymacji częstości etykietowania': (0, 0.5),
    r'Błąd estymacji prawdopodobieństwa a priori': (0, 0.5),
    'AUC': (None, None),
    'Czas wykonania': (None, None),
    'Iteracje metody': (None, None),
    'Ewaluacje funkcji w trakcie optymalizacji': (None, None),
}

metric_short_name = {
    'Błąd aproksymacji (AE) prawdopodobieństwa a posteriori': 'AE',
    r'Błąd estymacji częstości etykietowania': 'LFE',
    r'Błąd estymacji prawdopodobieństwa a priori': 'PPE',
    'AUC': 'AUC',
    'Czas wykonania': 'time',
    'Iteracje metody': 'it',
    'Ewaluacje funkcji w trakcie optymalizacji': 'ev',
}

best_function = {
    'Błąd aproksymacji (AE) prawdopodobieństwa a posteriori': np.min,
    r'Błąd estymacji częstości etykietowania': np.min,
    r'Błąd estymacji prawdopodobieństwa a priori': np.min,
    'AUC': np.max,
    'Czas wykonania': np.min,
    'Iteracje metody': np.min,
    'Ewaluacje funkcji w trakcie optymalizacji': np.min,
}

marker_styles = {
    'Naive - TIcE': {
        'color': 'brown',
        'marker': 'o',
        'fillstyle': 'full'
    },
    'Naive - EN': {
        'color': 'brown',
        'marker': 'o',
        'fillstyle': 'none'
    },
    'Weighted - TIcE': {
        'color': 'gray',
        'marker': '^',
        'fillstyle': 'full'
    },
    'Weighted - EN': {
        'color': 'gray',
        'marker': '^',
        'fillstyle': 'none'
    },
    'Ward - TIcE': {
        'color': 'gray',
        'marker': '^',
        'fillstyle': 'full'
    },
    'Ward - EN': {
        'color': 'gray',
        'marker': '^',
        'fillstyle': 'none'
    },
    'Joint': {
        'color': 'black',
        'marker': 's',
        'fillstyle': 'none'
    },
    'MM': {
        'color': palette[0],
        'marker': 'h',
        'fillstyle': 'none'
    },
    'Simple split': {
        'color': palette[1],
        'marker': '*',
        'fillstyle': 'none'
    },
    'DCCP': {
        'color': palette[2],
        'marker': 'X',
        'fillstyle': 'none'
    },
    'CCCP': {
        'color': palette[3],
        'marker': 'D',
        'fillstyle': 'none'
    },
    'Oracle': {
        'color': palette[4],
        'marker': None,
        'fillstyle': 'none'
    },
}

marker_styles['Ward'] = marker_styles['Ward - EN']
marker_styles['Weighted'] = marker_styles['Weighted - EN']
marker_styles['EN'] = marker_styles['Weighted - EN']
marker_styles['TIcE'] = marker_styles['Weighted - TIcE']
marker_styles['Naive'] = marker_styles['Naive - EN']

draw_order = [
    'Naive',
    'Naive - EN',
    'Naive - TIcE',
    'Weighted',
    'Weighted - EN',
    'Weighted - TIcE',
    'Weighted (EN)',
    'Weighted (TIcE)',
    'EN',
    'TIcE',
    'Ward',
    'Ward - EN',
    'Ward - TIcE',
    'Joint',
    'Simple split',
    'DCCP',
    'MM',
    'CCCP',
    'Oracle'
]
