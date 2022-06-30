import datetime, json, glob


def get_timestamp(strftime="%y%m%dT%H%M%S"):
    x = datetime.datetime.now()
    return x.strftime(strftime)


def save_as_json(d:dict, path):
    path = path + (".json" if path[-5:] != ".json" else "")
    with open(path, 'w') as file:
        json.dump(d, file)
        print(f'saving json at: {path}')


def get_path_from_re(re):
    return sorted(glob.glob(re))[-1]
