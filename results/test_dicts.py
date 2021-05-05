import numpy as np


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
    'Joint': {
        'color': 'black',
        'marker': 's',
        'fillstyle': 'none'
    },
    'CCCP': {
        'color': 'red',
        'marker': 'D',
        'fillstyle': 'none'
    },
    'MM': {
        'color': 'blue',
        'marker': 'h',
        'fillstyle': 'none'
    },
    'DCCP': {
        'color': 'green',
        'marker': 'X',
        'fillstyle': 'none'
    },
}

marker_styles['Weighted'] = marker_styles['Weighted - EN']
marker_styles['Naive'] = marker_styles['Naive - EN']

draw_order = [
    'Naive',
    'Naive - EN',
    'Naive - TIcE',
    'Weighted',
    'Weighted - EN',
    'Weighted - TIcE',
    'Joint',
    'DCCP',
    'MM',
    'CCCP',
]
