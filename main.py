import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from src.utils import setup_logger
from src.config import ensure_dirs

logger = setup_logger("Main")

STAGES = {
    'prepare': 'Подготовка данных',
    'train_ml': 'Обучение классической модели',
    'train_gnn': 'Обучение GNN-модели',
    'evaluate': 'Оценка метрик',
    'predict': 'Предсказание на новых данных',
    'segment': 'Сегментация DXF с визуализацией',
    'visualize': 'Визуализация результатов',
    'generate_revit': 'Генерация команд для Revit (JSON для Dynamo)',
    'segment_gnn': 'GNN сегментация плана этажа по слоям',
    'parse_plan':   'Парсинг плана этажа по слоям + визуализация',
    'generate_ifc':  'Генерация IFC файла для открытия в Revit',
    'compare_methods': 'Сравнение Rule-based vs ML vs GNN',
    'train_gnn_plans':  'Обучение GNN на реальных планах из pred_raw',
    'all': 'Запустить всё последовательно'
}

def run_stage(stage: str, **kwargs):
    if stage == 'prepare':
        from src.preprocessor import run_preprocessing
        run_preprocessing(**kwargs)
    elif stage == 'train_ml':
        from src.models.train_ml import train_ml_model
        train_ml_model(**kwargs)
    elif stage == 'train_gnn':
        from src.models.train_gnn import train_gnn_model
        train_gnn_model(**kwargs)
    elif stage == 'evaluate':
        sys.path.insert(0, str(Path(__file__).parent))
        from evaluation.evaluate import evaluate_ml_model
        evaluate_ml_model(**kwargs)
    elif stage in ['predict', 'segment']:
        from src.predict import predict_directory
        predict_directory(**kwargs)
    elif stage == 'visualize':
        sys.path.insert(0, str(Path(__file__).parent))
        from evaluation.visualize import plot_training_history, show_segmentation_results
        plot_training_history(history=[])
        show_segmentation_results()
    elif stage == 'all':
        logger.info("🚀 Running full pipeline...")
        for s in ['prepare', 'train_ml', 'train_gnn', 'evaluate', 'predict']:
            logger.info(f"\n>>> Stage: {s}")
            run_stage(s)
    elif stage == 'train_gnn_plans':
        from src.models.train_gnn_from_plans import train_gnn_on_plans
        train_gnn_on_plans(
            dxf_dir=kwargs.get('dir_path'),
            output_model_path=kwargs.get('output_model'),
            output_eval_dir=kwargs.get('output_dir'),
            aug_factor=kwargs.get('aug_factor', 8),
            epochs=kwargs.get('epochs', 100),
        )
    elif stage == 'compare_methods':
        from src.comparison_report import run_comparison
        run_comparison(
            dir_path=kwargs.get('dir_path'),
            output_dir=kwargs.get('output_dir'),
        )
    elif stage == 'generate_ifc':
        from src.ifc_generator import generate_ifc_directory
        generate_ifc_directory(
            dir_path=kwargs.get('dir_path'),
            output_dir=kwargs.get('output_dir'),
        )
    elif stage == 'parse_plan':
        from src.visualize_plan import visualize_directory
        visualize_directory(
            dir_path=kwargs.get('dir_path'),
            output_dir=kwargs.get('output_dir'),
        )
    elif stage == 'segment_gnn':
        from src.predict_layers import predict_directory_by_layers
        predict_directory_by_layers(
            dir_path=kwargs.get('dir_path'),
            gnn_model_path=kwargs.get('gnn_model_path'),
            output_dir=kwargs.get('output_dir'),
        )
    elif stage == 'generate_revit':
        from src.revit_generator import generate_for_directory
        generate_for_directory(**kwargs)
    else:
        logger.error(f"Unknown stage: {stage}")

def main():
    parser = argparse.ArgumentParser(description='DXF Object Recognition Pipeline')
    parser.add_argument('--stage', type=str, default='prepare')
    parser.add_argument('--input', type=str, default=None, help='Путь к DXF файлу или папке')
    parser.add_argument('--output', type=str, default=None, help='Путь для результата')
    parser.add_argument('--min-samples', type=int, default=None)
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--aug-factor', type=int, default=None)
    parser.add_argument('--no-revit-structure', action='store_true')
    parser.add_argument('--model', type=str, default='random_forest')
    parser.add_argument('--exclude-folders', type=str, nargs='+', default=None)
    parser.add_argument('--ignore-folders', type=str, nargs='+', default=None)
    args = parser.parse_args()

    ensure_dirs()

    if args.stage not in STAGES:
        logger.error(f"Неизвестный этап: {args.stage}")
        return

    logger.info(f"Запуск этапа: {args.stage}")

    kwargs = {}
    if args.stage == 'prepare':
        kwargs = {
            'min_samples_per_class': args.min_samples,
            'use_revit_structure': not args.no_revit_structure,
            'augment_data': args.augment,
            'augmentation_factor': args.aug_factor,
            'exclude_folders': args.exclude_folders,
            'ignored_folder_names': args.ignore_folders
        }
    elif args.stage == 'train_ml':
        kwargs = {'model_type': args.model}
    elif args.stage in ['predict', 'segment']:
        kwargs = {
            'dir_path': args.input,
            'output_dir': args.output,
            'model_path': None
        }
    elif args.stage == 'train_gnn_plans':
        kwargs = {
            'dir_path':   args.input,
            'output_dir': args.output,
        }
    elif args.stage == 'compare_methods':
        kwargs = {
            'dir_path':   args.input,
            'output_dir': args.output,
        }
    elif args.stage == 'generate_ifc':
        kwargs = {
            'dir_path':   args.input,
            'output_dir': args.output,
        }
    elif args.stage == 'parse_plan':
        kwargs = {
            'dir_path':   args.input,
            'output_dir': args.output,
        }
    elif args.stage == 'segment_gnn':
        kwargs = {
            'dir_path':        args.input,
            'gnn_model_path':  None,
            'output_dir':      args.output,
        }
    elif args.stage == 'generate_revit':
        kwargs = {
            'dxf_dir':         args.input,
            'predictions_dir': None,
            'output_dir':      args.output,
        }

    run_stage(args.stage, **kwargs)

if __name__ == "__main__":
    main()