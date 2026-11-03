import clr
clr.AddReference('RevitAPI')
clr.AddReference('RevitAPIUI')
clr.AddReference('RevitServices')
clr.AddReference('RevitNodes')

from Autodesk.Revit.DB import *
from RevitServices.Persistence import DocumentManager
from RevitServices.Transactions import TransactionManager
import System
from System.Collections.Generic import List
import os

doc = DocumentManager.Instance.CurrentDBDocument

# Входные данные: путь для экспорта (IN[0])
if isinstance(IN[0], str):
    export_folder = IN[0]
else:
    export_folder = UnwrapElement(IN[0])

# 1. СОБИРАЕМ ВСЕ СЕМЕЙСТВА ИЗ ПРОЕКТА
collector = FilteredElementCollector(doc).OfClass(FamilySymbol)
all_family_symbols = collector.ToElements()

# Фильтруем: убираем пустые и системные семейства (у них нет Category)
valid_symbols = []
for sym in all_family_symbols:
    if sym and sym.Category and not sym.IsSystemFamily:
        valid_symbols.append(sym)

# Убираем дубликаты по Id
seen = set()
unique_symbols = []
for sym in valid_symbols:
    if sym.Id.IntegerValue not in seen:
        seen.add(sym.Id.IntegerValue)
        unique_symbols.append(sym)

output_files = []

# Функция создания легенды
def create_or_get_legend(family_name):
    # Проверяем, существует ли уже легенда
    collector = FilteredElementCollector(doc).OfClass(View)
    for view in collector:
        if view.ViewType == ViewType.Legend and view.Name == family_name:
            return view
    # Создаём новую
    legend_view = View.CreateLegend(doc, None)
    legend_view.Name = family_name
    legend_view.Scale = 50
    return legend_view

# Функция очистки имени от недопустимых символов
def clean_filename(name):
    invalid_chars = '<>:"/\\|?*'
    for c in invalid_chars:
        name = name.replace(c, '_')
    return name.strip()

TransactionManager.Instance.EnsureInTransaction(doc)

for sym in unique_symbols:
    # Формируем имя
    if sym.Family:
        family_part = sym.Family.Name
    else:
        family_part = "UnknownFamily"
    
    type_part = sym.Name
    family_name = "{} - {}".format(family_part, type_part)
    
    safe_name = clean_filename(family_name)
    
    # Ограничиваем длину имени (Windows: 255 символов)
    if len(safe_name) > 200:
        safe_name = safe_name[:200]
    
    # Создаём или получаем легенду
    legend = create_or_get_legend(safe_name)
    
    # Очищаем легенду от старых элементов (если перезапускаем скрипт)
    # Получаем все элементы в легенде
    legend_collector = FilteredElementCollector(doc, legend.Id)
    existing_elements = legend_collector.ToElementIds()
    if existing_elements:
        doc.Delete(existing_elements)
    
    # Рисуем прямоугольник 10x10 мм (в метрах 0.01)
    pt1 = XYZ(0, 0, 0)
    pt2 = XYZ(0.01, 0.01, 0)
    
    # Создаём 4 линии прямоугольника
    line1 = Line.CreateBound(pt1, XYZ(pt2.X, pt1.Y, 0))
    line2 = Line.CreateBound(XYZ(pt2.X, pt1.Y, 0), pt2)
    line3 = Line.CreateBound(pt2, XYZ(pt1.X, pt2.Y, 0))
    line4 = Line.CreateBound(XYZ(pt1.X, pt2.Y, 0), pt1)
    
    DetailLine.Create(doc, legend.Id, line1)
    DetailLine.Create(doc, legend.Id, line2)
    DetailLine.Create(doc, legend.Id, line3)
    DetailLine.Create(doc, legend.Id, line4)
    
    # Добавляем текст (имя семейства) справа от прямоугольника
    text_origin = XYZ(0.015, 0.005, 0)  # на 5 мм правее
    TextNote.Create(doc, legend.Id, text_origin, family_name, 2, 
                   TextAlignFlags.Left, TextFontFlags.Bold)
    
    # Экспорт в DXF
    dxf_path = os.path.join(export_folder, safe_name + ".dxf")
    
    # Настройки экспорта
    dwg_options = DWGExportOptions()
    dwg_options.ExportViewsInSheetFormat = False
    
    views = List[ElementId]([legend.Id])
    
    # Экспортируем
    doc.Export(export_folder, safe_name + ".dxf", views, dwg_options)
    output_files.append(dxf_path)
    
    print("Создано: {}".format(safe_name))

TransactionManager.Instance.TransactionTaskDone()

# Выводим список созданных файлов
OUT = output_files