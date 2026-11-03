from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = Path("D:/RevitData")
TRAIN_RAW_DIR = DATA_DIR / "TrainFor2DFromRevit"
REVIT_LIBRARY_PATH = Path("C:/ProgramData/Autodesk/RVT 2024/Libraries/Russian/Russia")

PROCESSED_DIR = BASE_DIR / "data" / "processed"
PRED_RAW_DIR = PROCESSED_DIR / "pred_raw"
ML_OUTPUT_DIR = PROCESSED_DIR / "ml"
GNN_OUTPUT_DIR = PROCESSED_DIR / "gnn"
MODELS_DIR = PROCESSED_DIR / "models"
EVALUATION_DIR = PROCESSED_DIR / "evaluation"

TARGET_ENTITY_TYPES = ['LINE', 'CIRCLE', 'ARC', 'POLYLINE', 'LWPOLYLINE']
IGNORE_EXTENSIONS = ['.pcp', '.txt', '.log', '.dxf.lock']

MODEL_TYPE = 'random_forest'
GNN_MODEL_TYPE = 'gcn'
USE_GRID_SEARCH = False

# --- Пороги геометрии (миллиметры) ---
DEPTH_WALL_THRESHOLD  = 1500.0
DEPTH_SLAB_THRESHOLD  = 500.0
HEIGHT_WALL_THRESHOLD = 2500.0

# --- Веса классов для Random Forest ---
# Проблема: при весах 4.0 для малых классов (1 файл) модель начинала
# предсказывать их везде — перекос в другую сторону.
# Рекомендуемый диапазон подбора:
#   Классы с 8-10 файлами (Стены, Перекрытия, Потолки): 1.0 — 2.5
#   Классы с 1 файлом (Дверь, Окно, Вентшахта): 1.5 — 3.0
# Не поднимай малые классы выше 3.0 — при 1 файле это даёт перекос.
CLASS_WEIGHT_MULTIPLIERS = {
    'Стены':          1.0,
    'Перекрытия':     1.0,
    'Потолки':        1.0,
    'Базовая дверь':  1.0,
    'Базовое окно':   1.0,
    'Лестницы':       1.0,
    # Вентшахта убрана — мало данных, незначимый класс
}

# --- Настройки балансировки на уровне entity-строк в CSV ---
BALANCING_CONFIG = {
    # row_level: балансировка ПОСЛЕ извлечения признаков, по строкам CSV
    # Крупные классы режутся по файлам целиком (геометрия сохраняется)
    'strategy': 'row_level',
    'target_rows_per_class': None,   # None = авто (медиана по классам)
    'max_oversample_factor': 3.0,
}

# --- Настройки фильтрации и аугментации DXF-файлов ---
CLASS_FILTER_CONFIG = {
    # ВАЖНО: min_samples_per_class = 1 — иначе Дверь/Окно/Вентшахта (1 файл)
    # будут отфильтрованы до обучения. Несмотря на малый объём,
    # мы включаем их и компенсируем высокими весами классов.
    'min_samples_per_class': 1,
    'max_samples_per_class': None,
    'target_samples_per_class': 150,

    'use_revit_structure': True,
    'augment_data': True,

    # Высокий фактор аугментации для малых классов (Дверь, Окно = 1 файл)
    # После x15 у каждого будет ~15 файлов — уже можно учиться
    'augmentation_factor': 15,

    'exclude_folders': [
        'Вложенные семейства',
        'Common', 'Общие',
        'Templates', 'Шаблоны',
    ],
    'ignored_folder_names': [
        'Train', 'DXF', 'Data',
        'ГОСТ', 'GOST',
    ],

    'path_class_mapping': {
        'базовая дверь':  'Базовая дверь',
        'базовое окно':   'Базовое окно',
        'перекрытия':     'Перекрытия',
        'потолки':        'Потолки',
        'стены':          'Стены',
        'лестниц':        'Лестницы',   # лестницы / лестница
        # Вентшахта убрана — недостаточно данных
    },

    'include_classes': [
        'Стены',
        'Перекрытия',
        'Потолки',
        'Базовая дверь',
        'Базовое окно',
        'Лестницы',
        # Вентшахта убрана
    ],
}

# --- Настройки обучения GNN ---
# Были отсутствующими — из-за этого train_gnn падал с ImportError
GNN_TRAIN_CONFIG = {
    'hidden_channels': 64,
    'num_layers':      3,
    'lr':              0.005,
    'epochs':          80,
    'batch_size':      8,    # маленький batch — датасет небольшой
    'val_ratio':       0.2,
    'model_type':      'gcn',  # gcn | gat | graphsage | gin
}

# --- Grid Search для GNN (используется если USE_GRID_SEARCH=True) ---
GNN_GRID_SEARCH_CONFIG = {
    'epochs':   20,
    'n_folds':  3,
    'param_grid': {
        'hidden_channels': [32, 64, 128],
        'num_layers':      [2, 3],
        'lr':              [0.01, 0.005, 0.001],
    }
}

# --- Настройки генерации Revit ---
REVIT_GENERATOR_CONFIG = {
    'output_dir': str(PROCESSED_DIR / "revit_commands"),

    'default_wall_width':       0.2,
    'default_wall_height':      3.0,
    'default_column_height':    3.0,
    'default_floor_thickness':  0.2,

    'default_level_name': 'Level 1',
    'min_wall_length':    0.1,

    'class_to_family_mapping': {
        'Стены':          'Basic Wall',
        'Перекрытия':     'Basic Floor',
        'Потолки':        'Basic Ceiling',
        'Базовая дверь':  'M_Door',
        'Базовое окно':   'M_Window',
        'Вентшахта':      'Rectangular Shaft',
    }
}


def ensure_dirs():
    ML_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    GNN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    PRED_RAW_DIR.mkdir(parents=True, exist_ok=True)
