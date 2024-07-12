import numpy as np 
import cv2 
import os 

def save_results(folder_name, seq_name, results, data_type='default'):
    """
    write results to txt file

    results: list  row format: frame id, target id, box coordinate, class(optional)
    to_file: file path(optional)
    data_type: 'default' | 'mot_challenge', write data format, default or MOT submission
    """
    assert len(results)

    if not os.path.exists(f'./track_results/{folder_name}'):

        os.makedirs(f'./track_results/{folder_name}')

    with open(os.path.join('./track_results', folder_name, seq_name + '.txt'), 'w') as f:
        for frame_id, target_ids, tlwhs, clses, scores in results:
            for id, tlwh, score in zip(target_ids, tlwhs, scores):
                    f.write(f'{frame_id},{id},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{score:.2f},-1,-1,-1\n')

    f.close()

    return folder_name
