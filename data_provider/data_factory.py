from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Solar, Dataset_PEMS, \
    Dataset_Pred,Dataset_CEMP
from torch.utils.data import DataLoader
from data_provider.data_loader_calendar import Dataset_Solar_Calendar

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'Solar': Dataset_Solar,
    'SolarCalendar': Dataset_Solar_Calendar,
    'PEMS': Dataset_PEMS,
    'custom': Dataset_Custom,
    'CEMP': Dataset_CEMP,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    
    if flag in ('val', 'test'):
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    use_long_spectrum = int(getattr(args, "use_long_spectrum", 0)) == 1
    if use_long_spectrum:
        raise NotImplementedError("Long spectrum not supported in this backup")

    data_kwargs = dict(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
    )
    if args.data == 'SolarCalendar':
        data_kwargs["period"] = int(getattr(args, "sde_cycle_len", 144))
    if use_long_spectrum and flag != 'pred' and args.data in long_data_dict:
        data_kwargs["long_spectrum_factor"] = int(getattr(args, "long_spectrum_factor", 4))

    data_set = Data(**data_kwargs)
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
