from nuscenes import NuScenes
import json
import os.path as osp
import subprocess


CHOSEN_SCENE_INDICES = [0]
CHOSEN_SCENE_TOKENS = ['ce0d35zgspb9w90ytv0x9ik8bb6a1z7h']
DATA_ROOT = '../data/v2x-sim/v1.0-mini'
JSON_OUT = '../data/v2x-sim/v1.0-small'


def filter_scene():
    with open(osp.join(DATA_ROOT, 'scene.json')) as f:
        data = json.load(f)
    remove_indices = [idx for idx, scene in enumerate(data) if scene['token'] not in CHOSEN_SCENE_TOKENS]
    for idx in reversed(remove_indices):
        del data[idx]

    with open(osp.join(JSON_OUT, 'scene.json'), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def filter_log():
    with open(osp.join(JSON_OUT, 'scene.json')) as f:
        scenes = json.load(f)

    kept_log_tokens = [scene['log_token'] for scene in scenes]

    with open(osp.join(DATA_ROOT, 'log.json')) as f:
        logs = json.load(f)
    remove_log_indices = [idx for idx, log in enumerate(logs) if log['token'] not in kept_log_tokens]
    for idx in reversed(remove_log_indices):
        del logs[idx]

    with open(osp.join(JSON_OUT, 'log.json'), 'w', encoding='utf-8') as f:
        json.dump(logs, f, ensure_ascii=False, indent=4)


def filter_sample():
    with open(osp.join(DATA_ROOT, 'sample.json')) as f:
        samples = json.load(f)
    remove_sample_indices = [idx for idx, sample in enumerate(samples)
                             if sample['scene_token'] not in CHOSEN_SCENE_TOKENS]
    for idx in reversed(remove_sample_indices):
        del samples[idx]

    with open(osp.join(JSON_OUT, 'sample.json'), 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=4)


def filter_sample_data():
    with open(osp.join(JSON_OUT, 'sample.json')) as f:
        samples = json.load(f)

    chosen_sample_tokens = [sample['token'] for sample in samples]

    with open(osp.join(DATA_ROOT, 'sample_data.json')) as f:
        sample_datas = json.load(f)
    remove_sd_indices = [idx for idx, sd in enumerate(sample_datas)
                             if sd['sample_token'] not in chosen_sample_tokens]
    for idx in reversed(remove_sd_indices):
        del sample_datas[idx]

    with open(osp.join(JSON_OUT, 'sample_data.json'), 'w', encoding='utf-8') as f:
        json.dump(sample_datas, f, ensure_ascii=False, indent=4)


def filter_sample_annotation():
    with open(osp.join(JSON_OUT, 'sample.json')) as f:
        samples = json.load(f)

    chosen_sample_tokens = [sample['token'] for sample in samples]

    with open(osp.join(DATA_ROOT, 'sample_annotation.json')) as f:
        sample_annos = json.load(f)
    remove_sd_indices = [idx for idx, anno in enumerate(sample_annos)
                             if anno['sample_token'] not in chosen_sample_tokens]
    for idx in reversed(remove_sd_indices):
        del sample_annos[idx]

    with open(osp.join(JSON_OUT, 'sample_annotation.json'), 'w', encoding='utf-8') as f:
        json.dump(sample_annos, f, ensure_ascii=False, indent=4)


def copy_unfiltered_json_files():
    unfiltered_files = ['attribute.json', 'calibrated_sensor.json', 'category.json', 'ego_pose.json', 'instance.json',
                        'lidarseg.json', 'map.json', 'sensor.json', 'visibility.json']
    for filename in unfiltered_files:
        cmd_out = subprocess.run(['cp', osp.join(DATA_ROOT, filename), osp.join(JSON_OUT, filename)],
                                 stdout=subprocess.PIPE)


if __name__ == '__main__':
    # filter_scene()
    # filter_log()
    # filter_sample()
    # filter_sample_data()
    # filter_sample_annotation()
    copy_unfiltered_json_files()
