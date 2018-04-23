import argparse
import logging
import yaml
import datetime
import os

from catwalk.db import connect
from catwalk.storage import FSModelStorageEngine
from triage.experiments.multicore import MultiCoreExperiment

PROJECT_PATH = 'experiment_output/'

def run(verbose, config_filename, features_directory, replace):
    # configure logging
    log_filename = 'logs/modeling_{}'.format(
        str(datetime.datetime.now()).replace(' ', '_').replace(':', '')
    )
    if verbose:
        logging_level = logging.DEBUG
    else:
        logging_level = logging.INFO
    logging.basicConfig(
        format='%(asctime)s %(process)d %(levelname)s: %(message)s', 
        level=logging_level,
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()]
    )

    # load main experiment config
    with open(config_filename) as f:
        experiment_config = yaml.load(f)

    # load feature configs and update experiment config with their contents
    all_feature_aggregations = []
    for filename in os.listdir('config/{}/'.format(features_directory)):
        with open('config/{}/{}'.format(features_directory, filename)) as f:
            feature_aggregations = yaml.load(f)
            for aggregation in feature_aggregations:
                all_feature_aggregations.append(aggregation)
    experiment_config['feature_aggregations'] = all_feature_aggregations
    
    db_engine = connect()
    pipeline = MultiCoreExperiment(
        n_processes=2,
        n_db_processes=2,
        config=experiment_config,
        db_engine=db_engine,
        model_storage_class=FSModelStorageEngine,
        project_path=PROJECT_PATH,
        replace=False
    )

    try:
        pipeline.run()
    except Exception as e:
        logging.exception(e)
        raise e

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run triage pipeline')
    parser.add_argument(
        "-v",
        "--verbose",
        help="Enable debug logging",
        action="store_true"
    )
    parser.add_argument(
        "-c",
        "--config_filename",
        type=str,
        help="Pass the config filename"
    )
    parser.add_argument(
        "-f",
        "--features_directory",
        type=str,
        nargs='?',
        default='features',
        help="Pass the features directory; for most runs, this is 'features'"
    )
    parser.add_argument(
        "-r",
        "--replace",
        help="Pass the features directory; for most runs, this is 'features'",
        action="store_true"
    )

    args = parser.parse_args()

    run(args.verbose, args.config_filename, args.features_directory, args.replace)
