from nuscenes import NuScenes


THRESHOLD_DISTANCE_TO_LIDAR = 2.0


def get_a_sample(scene_idx: int = 0, sample_idx: int = 0):
    nusc = NuScenes(dataroot='../data/v2x-sim', version='v1.0-small', verbose=True)
    scene = nusc.scene[scene_idx]
    sample_token = scene['first_sample_token']
    for s_idx in range(sample_idx):
        sample_rec = nusc.get('sample', sample_token)
        sample_token = sample_rec['next']
    return nusc, sample_token


def print_record(record: dict, ignored_keys=('anns'), root=True):
    if root:
        print('---')
        print('{')

    for k, v in record.items():
        if k in ignored_keys:
            continue
        if k == 'data':
            print(f"{k}:")
            print_record(v, root=False)
        else:
            print(f'\t{k}: {v}')

    if root:
        print('}')
        print('---')

