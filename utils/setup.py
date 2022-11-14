from subprocess import Popen, PIPE


def check_if_there_is_uncommited_changes():
    process = Popen(["git", "status"], stdout=PIPE)
    (commit, err) = process.communicate()
    process.wait()
    commit = str(commit)

    if 'Changes not staged for commit' in commit:
        raise Exception('Please stage and commit branch before start your experiment!')
    if 'Changes to be committed' in commit:
        raise Exception('Please commit branch before start your experiment!')
    if 'Untracked files' in commit:
        raise Exception('Please commit untracked files before start your experiment!')

def setup_options(opt, dataset_name, dataset_config):
    try:
        opt.samples = dataset_config[dataset_name]['samples']
        opt.classes = dataset_config[dataset_name]['classes']
        opt.labels = dataset_config[dataset_name]['labels']
    
    except Exception as e:
        raise TypeError(f'Dataset {dataset_name} does not exist: {e}')
    print(opt)    
    return opt