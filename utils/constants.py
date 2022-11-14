

def get_dataset_config(opt):

    if opt.config_type == 1:
        return dataset_config

    elif opt.config_type == 2:
        return dataset_config_2





dataset_config = {
    'bciiv1': {
        'channels': 59,
        'samples': 200,
        'classes': 3,
        'labels':['Left hand', 'Right hand','Foot'],
        'eletrode_list':['all'],
        'eletrode_index':[],
        'pick_eletrodes':False,
    },
    'bciiv2a': {
        'channels': 22,
        'samples': 201,
        'classes': 4,
        'labels':['Left hand', 'Right hand','Foot','Tongue'],
        'eletrode_list':['all'],
        'eletrode_index':[],
        'pick_eletrodes':False,
    },
    'bciiv2a_mat': {
        'channels': 22,
        'samples': 1750,
        'classes': 4,
        'labels':['Left hand', 'Right hand','Foot','Tongue'],
        'eletrode_list':['all'],
        'eletrode_index':[],
        'pick_eletrodes':False,
    },
    'eegmi': {
        'channels': 64,
        'samples': 641,
        'classes': 4,
        'labels':['Left hand', 'Right hand','Both hands','Foot'],
        'eletrode_list':['all'],
        'eletrode_index':[],
        'pick_eletrodes':False,
    }
}


dataset_config_2 = {
    'bciiv1': {
        'channels': 7,
        'samples': 200,
        'classes': 3,
        'labels':['Left hand', 'Right hand','Foot'],
        'eletrode_list':['C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6'],
        'eletrode_index':[25, 26, 27, 28, 29, 30, 31],
        'pick_eletrodes':True,
    },
    'bciiv2a': {
        'channels': 7,
        'samples': 201,
        'classes': 4,
        'labels':['Left hand', 'Right hand','Foot','Tongue'],
        'eletrode_list':['C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6'],
        'eletrode_index':[7, 8, 9, 10, 11, 12, 13],
        'pick_eletrodes':True,
    },
    'bciiv2a_mat': {
        'channels': 7,
        'samples': 1750,
        'classes': 4,
        'labels':['Left hand', 'Right hand','Foot','Tongue'],
        'eletrode_list':['C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6'],
        'eletrode_index':[7, 8, 9, 10, 11, 12, 13],
        'pick_eletrodes':True,
    },
    'eegmi': {
        'channels': 7,
        'samples': 641,
        'classes': 4,
        'labels':['Left hand', 'Right hand','Both hands','Foot'],
        'eletrode_list':['C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6'],
        'eletrode_index':[8, 9, 10, 11, 12, 13, 14],
        'pick_eletrodes':True,
    },
    
}
