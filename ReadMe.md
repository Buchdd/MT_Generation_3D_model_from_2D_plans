1. python main.py --stage prepare
   → создаёт train_features.csv из DXF обучающих данных

2. python main.py --stage train_ml
   → обучает Random Forest, сохраняет ml_model.pkl

3. python main.py --stage train_gnn
   → обучает GNN, сохраняет gnn_model.pt

4. python main.py --stage parse_plan
   → парсит DXF из pred_raw по слоям
   → создаёт картинки *_parsed.png для проверки

5. python main.py --stage generate_revit
   → берёт результаты парсинга
   → генерирует JSON команды для Dynamo

6. Открываешь Revit → Dynamo → GenerateRevit.dyn
   → указываешь путь к папке revit_commands
   → нажимаешь Run
   → в Revit появляются стены, двери, окна


python main.py --stage prepare
python main.py --stage train_ml
python main.py --stage train_gnn
python main.py --stage parse_plan
python main.py --stage generate_revit
python main.py --stage generate_ifc 
python main.py --stage compare_methods