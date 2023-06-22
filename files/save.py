import os
import torch
import model_size

def save_model_w_condition(model, model_dir, model_name, accu, target_accu, log=print, state_dict=False):
    '''
    model: this is not the multigpu model
    '''
    if accu > target_accu:
        filename = os.path.join(model_dir, (model_name + '_' + '{0:.4f}'.format(accu).replace('.', '_') + '.pth'))
        log('\tabove {0:.2f}%'.format(target_accu * 100))

        if state_dict:
            filename = state_dict_path(filename)
            torch.save(obj=model.state_dict(), f=filename)
            # model_size.get_model_size(filename, log=log)
        else:
            torch.save(obj=model, f=filename)


def state_dict_path(model_path):
    '''
    model_path: the path of the model
    '''
    dir_name, file_name = os.path.split(model_path)
    base_name, extension = os.path.splitext(model_path)
    new_base_name = base_name + '_state_dict'
    new_file_path = os.path.join(dir_name, new_base_name + extension)

    return new_file_path


if __name__ == '__main__':
    model_path = 'C:\\Users\\dilsh\\OneDrive\\Documents\\Dilshaan\\Third_Year\\BEng_Individual_Project\\code\\BEng-Final-Year-Project\\model\\.pth'
    print(state_dict_path(model_path))
