import logging
from pathlib import Path
import re

# Игнорируемые расширения
IGNORE_EXTENSIONS = ['.pcp', '.txt', '.log', '.dxf.lock']

def setup_logger(name, log_file=None, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger

def get_dxf_files(directory: Path, recursive: bool = False):
    """Находит все DXF файлы в директории"""
    if recursive:
        return list(directory.rglob("*.dxf"))
    else:
        return [f for f in directory.iterdir() if f.suffix.lower() == '.dxf']

def extract_class_from_filename(filename: str) -> str:
    """
    Извлекает класс из имени файла (без расширения).
    Убирает .dxf и суффиксы аугментации.
    """
    name = Path(filename).stem
    
    # Убираем суффиксы аугментации (_aug0, _aug1, и т.д.)
    name = re.sub(r'_aug\d+$', '', name)
    
    # Убираем дополнительные суффиксы (-Колонна, -01, и т.д.)
    name = re.sub(r'[-_]\d+$', '', name)
    name = re.sub(r'-Колонна$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'-01$', '', name)
    
    return name.strip()


def get_class_statistics(dxf_files, class_extractor=None):
    """Статистика по классам для проверки перед обучением"""
    from collections import Counter
    
    if class_extractor:
        classes = [class_extractor(f) for f in dxf_files]
    else:
        classes = [extract_class_from_filename(f.name) for f in dxf_files]
    
    class_counts = Counter(classes)
    
    print("\n" + "="*60)
    print("📊 СТАТИСТИКА КЛАССОВ")
    print("="*60)
    print(f"Всего файлов: {len(dxf_files)}")
    print(f"Всего классов: {len(class_counts)}")
    if len(class_counts) > 0:
        print(f"Среднее файлов на класс: {len(dxf_files)/len(class_counts):.1f}")
    print("\nТоп-20 классов:")
    
    for class_name, count in sorted(class_counts.items(), key=lambda x: -x[1])[:20]:
        bar = '█' * min(count, 50)
        print(f"  {class_name:40} | {bar} ({count})")
    
    # Проверка на классы с 1 образцом
    singletons = [c for c, n in class_counts.items() if n < 2]
    if singletons:
        print(f"\n⚠️  ПРЕДУПРЕЖДЕНИЕ: {len(singletons)} классов имеют < 2 образцов!")
    
    print("="*60 + "\n")
    
    return class_counts