import numpy as np
import os
import torch
import sys
from ABE_M_model import ABE_M
from my_model import se_resnet101
from resnet import resnet101
from dataset import SingleData
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchsummary import summary


def get_data_list(data_path):
    img_list = []
    for root, dirs, files in os.walk(data_path):
        if files == []:
            class_name = dirs
        elif dirs == []:
            for f in files:
                img_path = os.path.join(root, f)
                img_list.append(img_path)
    return class_name, img_list


def test(test_loader, model, devicei):
    with torch.no_grad():
        model.eval()
        embedding_vec = []
        #test_img_list = []
        for batch_idx, (data, target, img_name) in enumerate(test_loader):
            data = data.to(device)
            outputs = model(data)
            #print(type(img_name)) 
            if batch_idx == 0:
                for i in range(len(outputs)):
                    output = outputs[i].cpu().numpy()
                    embedding_vec.append(output)
                labels = np.array(target)
                test_img_list = img_name
            else:
                for i in range(len(outputs)):
                    output = outputs[i].cpu().numpy()
                    embedding_vec[i] = np.concatenate((embedding_vec[i], output))
                labels = np.concatenate((labels, np.array(target)))
                test_img_list += img_name
    return embedding_vec, labels, list(test_img_list)


def Accuracy(embedding_vec, labels, test_img_list):
    print(labels)
    def _get_sim_matrix(embedding_vec, num_vec):
        sim_matrix = np.zeros((num_vec, num_vec))
        for i in range(len(embedding_vec)):
            sim_matrix += np.matmul(embedding_vec[i], embedding_vec[i].T)
        if len(embedding_vec) > 1:
            sim_matrix = sim_matrix / len(embedding_vec)
        return sim_matrix
    #k_list = [1,2,4,8]
    k_list = [1]
    max_k = max(k_list)
    num_vec = embedding_vec[0].shape[0]
    print('total num of test image: {}'.format(num_vec))
    sim_mat = _get_sim_matrix(embedding_vec, num_vec)
    result = np.zeros((num_vec, max_k+1))
    
    #f = open('result.txt', 'w')
    
    for i in range(num_vec):
        temp_index = np.argsort(-1*sim_mat[i])[:max_k+1]
        for j in range(max_k+1):
            result[i][j] = labels[temp_index[j]]
            #f.write(test_img_list[i]+' '+test_img_list[temp_index[j]]+' '+str(sim_mat[i][temp_index[j]])+'\n')
    #f.close() 
    predicts = result[:,1]
    n_p = sum(labels==predicts)/num_vec
    num_class = len(set(labels))
    cnf_mat = np.zeros((num_class, num_class))
    for i in range(len(labels)):
        cnf_mat[int(labels[i])][int(predicts[i])] += 1

    #np.save('./model/cnf_mat.npy', cnf_mat)

    e = cnf_mat.diagonal()
    n_w = (e/cnf_mat.sum(axis=1)).mean()
    n_total = n_w*n_p
    accuracy = [(n_p,'np'), (n_w, 'nw'), (n_total, 'n_total')]
    '''
    recall = []
    for k in k_list:
        total = 0
        for i in range(num_vec):
            temp_list = result[i][1:k+1].tolist()
            if labels[i] in temp_list:
                total += 1
        recall.append((total/num_vec, k))
        print('Recall@k:{:.4f}, k={}'.format(total/num_vec, k))
    '''
    return accuracy 


if __name__ == '__main__':
    arg_len = len(sys.argv)
    if arg_len != 2:
        raise Exception("Invalid argvs!")
    weight_folder = sys.argv[1]
    model_name = 'se_resnet'
    #weight_type = weight_folder.split('/')[-1]

    model_dict = {'ABE_M':ABE_M, 'se_resnet':se_resnet101, 'resnet101':resnet101}
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    #num_learner = 4
    
        
    #data
    transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor()
        ])

    data_path = './KIMIA_data/test/'
    class_name, img_list = get_data_list(data_path)
    test_dataset = SingleData(class_name, img_list, transform)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    f = open(weight_folder+model_name+'_'+'results.txt', 'w')
    
    sorted_weight_file = os.listdir(weight_folder)
    sorted_weight_file.sort()
    for weight_f in sorted_weight_file:
        if weight_f[-3:] != 'pth' or model_name not in weight_f:
            continue
        weight_path = os.path.join(weight_folder, weight_f)
        print(weight_path)

        model = model_dict[model_name]() #####################################
        model.load_state_dict(torch.load(weight_path))
        model.to(device)


        embedding_vec, labels, test_img_list = test(test_dataloader, model, device)
        #print(test_img_list)
        accuracy_his = Accuracy(embedding_vec, labels, test_img_list)
        print(accuracy_his)
        f.write(weight_f + str(accuracy_his)+'\n')

    f.close()

