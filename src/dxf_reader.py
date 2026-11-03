import ezdxf
from src.utils import setup_logger

logger = setup_logger("DXFReader")

def load_dxf_document(file_path):
    """Загружает DXF файл и возвращает документ и modelspace"""
    try:
        doc = ezdxf.readfile(str(file_path))
        msp = doc.modelspace()
        return doc, msp
    except Exception as e:
        logger.error(f"Failed to read {file_path}: {e}")
        return None, None

def get_entities(msp, entity_types=None):
    """Возвращает список сущностей нужных типов"""
    if entity_types is None:
        return list(msp)
    
    entities = []
    for entity in msp:
        if entity.dxftype() in entity_types:
            entities.append(entity)
    return entities