"""
Usage:
    imagenet_to_subset.py <split_root> <out_root> <sampling_rate>
"""
import docopt
import os


if __name__ == '__main__':
    args = docopt.docopt(__doc__)
    root = args['<split_root>']
    out_root = args['<out_root>']
    sampling_rate = float(args['<sampling_rate>'])

    if not os.path.exists(root):
        raise ValueError(f"Root {root} doesn't exist.")
    if os.path.exists(out_root):
        raise ValueError(f"Output directory {out_root} exist already.")

    if sampling_rate > 1.0:
        raise ValueError(f"Sampling rate cannot be higher than 1.")

    data_dir = os.path.join(root, 'data')
    idx_dir = os.path.join(root, 'idx_files')

    records = sorted(os.listdir(data_dir))
    idxs = sorted(os.listdir(idx_dir))
    print(len(records), len(idxs))

    if len(records) != len(idxs):
        raise ValueError(f"Inequal amount of records and index files.")

    stride = 1. / sampling_rate
    next = 0.0
    new_records = []
    new_idxs = []
    for i, (record, idx) in enumerate(zip(records, idxs)):
        # print(i, next, record, idx)
        if i < next:
            continue
        new_records.append(record)
        new_idxs.append(idx)
        next += stride

    print(f"Writing symlinks:")
    print(len(records), 'x', sampling_rate, '=', len(records) * sampling_rate, '=', len(new_records))

    # Write symlinks
    os.makedirs(out_root)
    os.makedirs(os.path.join(out_root, 'data'))
    os.makedirs(os.path.join(out_root, 'idx_files'))
    for record, idx in zip(new_records, new_idxs):
        old = os.path.join(os.getcwd(), root, 'data', record)
        new = os.path.join(os.getcwd(), out_root, 'data', record)
        os.symlink(old, new)

        old = os.path.join(os.getcwd(), root, 'idx_files', idx)
        new = os.path.join(os.getcwd(), out_root, 'idx_files', idx)
        os.symlink(old, new)