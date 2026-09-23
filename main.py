import json, time, torch, numpy as np
from model_arch import QPPNet
from dataset.terrier_tpch_dataset.terrier_utils import TerrierTPCHDataSet
from dataset.postgres_tpch_dataset.tpch_utils import PSQLTPCHDataSet
from dataset.oltp_dataset.oltp_utils import OLTPDataSet
from dataset.postgres_plan_dataset import PostgresPlanDataSet
import argparse


parser = argparse.ArgumentParser(description='QPPNet Arg Parser')

# Environment arguments
# required

parser.add_argument('--data_dir', type=str, default='./res_by_temp/',
                    help='Dir containing train data')

parser.add_argument('--dataset', type=str, default='PSQLTPCH',
                    help='Select dataset [PostgresPlan | PSQLTPCH | TerrierTPCH | OLTP]')

parser.add_argument('--split_seed', type=int, default=2027,
                    help='Template-family train/test split seed')
parser.add_argument('--split_mode', choices=('family', 'role'), default='family',
                    help='Split by template family or cached history/holdout role')

parser.add_argument('--test_time', action='store_true',
                    help='if in testing mode')

parser.add_argument('--device', default='auto',
                    help='PyTorch device [auto | cpu | mps | cuda]')

parser.add_argument('-dir', '--save_dir', type=str, default='./saved_model',
                    help='Dir to save model weights (default: ./saved_model)')

parser.add_argument('--lr', type=float, default=1e-3,
                    help='Learning rate (default: 1e-3)')

parser.add_argument('--scheduler', action='store_true')
parser.add_argument('--step_size', type=int, default=1000,
                    help='step_size for StepLR scheduler (default: 1000)')

parser.add_argument('--gamma', type=float, default=0.95,
                    help='gamma in Adam (default: 0.95)')

parser.add_argument('--SGD', action='store_true',
                    help='Use SGD as optimizer with momentum 0.9')



parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size used in training (default: 32)')

parser.add_argument('-s', '--start_epoch', type=int, default=0,
                    help='Epoch to start training with (default: 0)')

parser.add_argument('-t', '--end_epoch', type=int, default=200,
                    help='Epoch to end training (default: 200)')

parser.add_argument('-epoch_freq', '--save_latest_epoch_freq', type=int, default=100)

parser.add_argument('-logf', '--logfile', type=str, default='train_loss.txt')

parser.add_argument('--mean_range_dict', type=str)
parser.add_argument('--predictions', type=str,
                    help='Write held-out PostgreSQL predictions as JSONL')
parser.add_argument('--load_epoch', type=str,
                    help='Load a saved epoch (for example, best) before evaluation')


def write_predictions(path, model, dataset):
    slices = {
        str(record['key']): record.get('evaluation_slice', 'holdout')
        for record in getattr(dataset, 'test_records', ())
    }
    censored = {
        str(record['key']): bool(record.get('censored', False))
        for record in getattr(dataset, 'test_records', ())
    }
    with open(path, 'w', encoding='utf-8') as output:
        for key, actual, predicted in zip(
                model.last_query_keys, model.last_truths,
                model.last_predictions):
            output.write(json.dumps({
                'key': key, 'actual_ms': float(actual),
                'evaluation_slice': slices.get(key, 'holdout'),
                'predicted_ms': float(predicted),
                'censored': censored.get(key, False),
            }) + '\n')

def save_opt(opt, logf):
    """Print and save options
    It will print both current options and default values(if different).
    It will save options into a text file / [checkpoints_dir] / opt.txt
    """
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)
    logf.write(message)
    logf.write('\n')

if __name__ == '__main__':
    opt = parser.parse_args()
    np.random.seed(opt.split_seed)
    torch.manual_seed(opt.split_seed)

    if opt.dataset == "PostgresPlan":
        dataset = PostgresPlanDataSet(opt)
        opt.dim_dict = dataset.dim_dict
    elif opt.dataset == "PSQLTPCH":
        dataset = PSQLTPCHDataSet(opt)
    elif opt.dataset == "TerrierTPCH":
        dataset = TerrierTPCHDataSet(opt)
    else:
        dataset = OLTPDataSet(opt)

    print("dataset_size", dataset.datasize)
    torch.set_default_tensor_type(torch.FloatTensor)
    qpp = QPPNet(opt)

    total_iter = 0

    if opt.test_time:
        qpp.evaluate(dataset.test_dataset if opt.predictions else dataset.all_dataset)
        print('total_loss: {}; test_loss: {}; pred_err: {}; R(q): {}' \
              .format(qpp.last_total_loss, qpp.last_test_loss,
                      qpp.last_pred_err, qpp.last_rq))
    else:
        logf = open(opt.logfile, 'w+')
        save_opt(opt, logf)
        #qpp.test_dataset = dataset.create_test_data(opt)
        qpp.test_dataset = dataset.validation_dataset

        for epoch in range(opt.start_epoch, opt.end_epoch):
            epoch_start_time = time.time()  # timer for entire epoch
            iter_data_time = time.time()    # timer for data loading per iteration
            epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

            samp_dicts = dataset.sample_data()
            total_iter += opt.batch_size

            qpp.set_input(samp_dicts)
            qpp.optimize_parameters(
                epoch,
                evaluate=(epoch % 50 == 0 or epoch + 1 == opt.end_epoch),
            )
            logf.write("epoch: " + str(epoch) + "; iter_num: " + str(total_iter) \
                      + '; total_loss: {}; test_loss: {}; pred_err: {}; R(q): {}' \
                      .format(qpp.last_total_loss, qpp.last_test_loss,
                              qpp.last_pred_err, qpp.last_rq))

            #if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
            losses = qpp.get_current_losses()
            loss_str = "losses: "
            for op in losses:
              loss_str += str(op) + " [" + str(losses[op]) + "]; "

            if epoch % 50 == 0:
                print("epoch: " + str(epoch) + "; iter_num: " + str(total_iter) \
                      + '; total_loss: {}; test_loss: {}; pred_err: {}; R(q): {}' \
                      .format(qpp.last_total_loss, qpp.last_test_loss,
                              qpp.last_pred_err, qpp.last_rq))
                print(loss_str)


            logf.write(loss_str + '\n')

            if (epoch + 1) % opt.save_latest_epoch_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch + 1, total_iter))
                qpp.save_units(epoch + 1)

        logf.close()

    if opt.predictions and not opt.test_time:
        qpp.load('best')
        qpp.evaluate(dataset.test_dataset)
    if opt.predictions:
        write_predictions(opt.predictions, qpp, dataset)
