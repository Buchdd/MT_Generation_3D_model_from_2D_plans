import json
from pathlib import Path
data = json.load(open('data/processed/revit_commands/ExampleDWG1_revit_commands.json', encoding='utf-8'))
walls = [c for c in data['commands'] if c['method']=='wall'][:3]
for w in walls:
    print('start:', w['geometry']['start'])
    print('end:  ', w['geometry']['end'])
    print()