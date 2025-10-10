from config import parser
from utils.helper import set_up_logger, set_seed, set_device, load_config, get_dir_path, save_config
from trainers import _trainer_dict


def main():
    args = parser.parse_args()
    args = vars(args)
    args = load_config(args)

    saving_path, saving_name = set_up_logger(args)

    set_seed(args['train'].get('seed', 42))
    args['train']['saving_path'] = saving_path
    args['train']['saving_name'] = saving_name
    save_config(args, saving_path)
    
    trainer = _trainer_dict[args['model']['name']](args)
    trainer.process()


if __name__ == "__main__":
    main()
