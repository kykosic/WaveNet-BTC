"""
    Main execution for WaveNet forecasting
"""
import os
import argparse
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from src.util import load_data, split_data, smape, Normalizer
from src.model import WaveNet


def set_logging():
    """ Set logging level and format """
    fmt = "[%(levelname)s %(asctime)s] %(message)s"
    logging.basicConfig(format=fmt, level=logging.INFO)

def parse_args():
    """ Parse command line arguments """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--date',
        type=lambda s: datetime.strptime(s, '%Y-%m-%d'),
        help='Date to run the model on.',
        required=True
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='./data',
        help='Location of CSV files for data input. Default is `./data`'
    )
    parser.add_argument(
        '--test_periods',
        type=int,
        default=360,
        help='Number of periods to use for testing. Default is 360sec (6 hours).'
    )
    parser.add_argument(
        '--num_filters',
        type=int,
        default=1,
        help='Convolutional filters to use. Default is 1.'
    )
    parser.add_argument(
        '--num_layers',
        type=int,
        default=7,
        help='Number of layers to use. Default is 7.',
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-3,
        help='Optimizer learning rate. Default is 1e-3.',
    )
    parser.add_argument(
        '--regularization',
        type=float,
        default=1e-2,
        help='L2 regularization coefficient. Default is 1e-2.',
    )
    parser.add_argument(
        '--num_iters',
        type=int,
        default=8000,
        help='Number of training iterations to run. Default is 8000.',
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='./logs',
        help='Directory to store logs and TensorBoard checkpoints. Default is ./logs'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed to use for weight initialization. Default is 0'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Store a plot of the resulting prediction'
    )
    parser.add_argument(
        '--to_csv',
        action='store_true',
        help='Store a CSV file of the predictions.'
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    set_logging()
    args = parse_args()

    # Load in data
    train_features, train_targets, test_targets = (
        load_data(args.date, args.data_dir)
        .pipe(lambda df: split_data(df, args.test_periods))
    )
    logging.info("Train periods: %s, test periods: %s",
                 train_features.shape[0], test_targets.shape[0])

    # Normalize data
    normalizer = Normalizer()
    train_features = normalizer.fit_transform(train_features)
    train_targets = normalizer.transform(train_targets)
    test_targets = normalizer.transform(test_targets)

    # Format model training input
    columns = train_features.columns.tolist()
    features = dict()
    targets = dict()
    for column in columns:
        features[column] = train_features[column].values.reshape(1, -1)
        targets[column] = train_targets[column].values.reshape(1, -1)

    # Model parameters
    params = {
        'time_steps': train_features.shape[0],
        'num_filters': args.num_filters,
        'num_layers': args.num_layers,
        'learning_rate': args.learning_rate,
        'regularization': args.regularization,
        'num_iters': args.num_iters,
        'log_dir': args.log_dir,
        'columns': columns,
        'seed': args.seed
    }
    logging.info("Training with parameters: %s", params)

    # Run model
    with WaveNet(**params) as model:

        # Train
        train_pred = model.train(targets, features)

        # Generate
        num_steps = test_targets.shape[0]
        test_pred = model.generate(num_steps, features)

    for column in columns:
        train_smape = smape(train_pred[column], train_targets[column], normalizer)
        test_smape = smape(test_pred[column], test_targets[column], normalizer)
        logging.info("%s in-sample sMAPE: %f", column, train_smape)
        logging.info("%s, out-of-sample sMAPE: %f", column, test_smape)

    # Maybe plot
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.style.use('seaborn-darkgrid')

        # Each column gets it's own plot
        for column in columns:
            # Combine in-sample and out-of-sample in a single series
            actual = pd.Series(
                np.hstack([train_targets[column].values, test_targets[column].values]),
                index=np.hstack([train_targets.index, test_targets.index]))
            forecast = pd.Series(
                np.hstack([train_pred[column].reshape(-1), test_pred[column].reshape(-1)]),
                index=np.hstack([train_targets.index, test_targets.index]))

            # Inverse transform
            actual = normalizer.inverse_transform(actual)
            forecast = normalizer.inverse_transform(forecast)
            
            fig = plt.figure(figsize=(12, 8))
            actual.plot(label='Actual', ls='--')
            forecast.plot(label='Forecast', alpha=.75)
            plt.title("{:s} on {:%Y-%m-%d}".format(column, args.date), fontsize=14)
            train_cutoff = args.date + timedelta(days=1) - timedelta(seconds=60 * args.test_periods)
            plt.axvline(train_cutoff, ls='--', alpha=.6)
            plt.xlabel("Timestamp", fontsize=12)
            plt.ylabel("Price USD", fontsize=12)
            plt.legend(fontsize=12)
            figname = os.path.join(args.log_dir, "{:s}_{:%Y_%m_%d}.png".format(column, args.date))
            plt.savefig(figname)
            logging.info("Plot saved to %s", figname)

    # Maybe save csv
    if args.to_csv:
        actual = (
            pd.DataFrame({
                col: np.hstack([train_targets[col].values, test_targets[col].values])
                for col in columns
                }, index=np.hstack([train_targets.index, test_targets.index]))
            .pipe(normalizer.inverse_transform)
        )
        forecast = (
            pd.DataFrame({
                col + '_pred': np.hstack([train_pred[col].reshape(-1), test_pred[col].reshape(-1)])
                for col in columns
                }, index=np.hstack([train_targets.index, test_targets.index]))
            .pipe(normalizer.inverse_transform)
        )
        filename = os.path.join(args.log_dir, "{:%Y_%m_%d}.csv".format(args.date))
        df = actual.join(forecast)
        df.index.name = 'Timestamp'
        df.to_csv(filename)
        logging.info("Output saved to %s", filename)
