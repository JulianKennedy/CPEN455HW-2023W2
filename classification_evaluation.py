'''
This code is used to evaluate the classification accuracy of the trained model.
You should at least guarantee this code can run without any error on validation set.
And whether this code can run is the most important factor for grading.
We provide the remaining code, all you should do are, and you can't modify the remaining code:
1. Replace the random classifier with your trained model.(line 64-68)
2. modify the get_label function to get the predicted label.(line 18-24)(just like Leetcode solutions)
'''
from torchvision import datasets, transforms
from utils import *
from model import * 
from dataset import *
from tqdm import tqdm
from pprint import pprint
import argparse
NUM_CLASSES = len(my_bidict)

# Write your code here
# And get the predicted label, which is a tensor of shape (batch_size,)


def get_label(model, model_input, device):
    #Begin of your code
    # run th model on the input and get the predicted label
    # call the discretized logistic mixture loss image
    # and return the predicted label

    # run the model through the forward pass
    # get the logits
    # get the predicted label
    # return the predicted label
    # make the labels

    labels = [0,1,2,3]
    labels = torch.tensor(labels, dtype=torch.int64).to(device)

    # get the logits
    logits = None
    for i in range(4):
        #create a labels tensor with size model_input[0]
        labels_tensor = torch.tensor([i]*model_input.shape[0], dtype=torch.int64).to(device)
        # print(labels_tensor.shape)
        if i == 0:
            logits = model(model_input, labels_tensor)
            image_loss = discretized_mix_logistic_loss_image(model_input, logits).unsqueeze(1)
            
        else:
            logits_1 = model(model_input, labels_tensor)
            image_loss = torch.cat((image_loss, discretized_mix_logistic_loss_image(model_input, logits_1).unsqueeze(1)), dim=1)
    
    predicted_label  = torch.argmin(image_loss, dim=1)

    return predicted_label





# End of your code

def classifier(model, data_loader, device):
    model.eval()
    acc_tracker = ratio_tracker()
    for batch_idx, item in enumerate(tqdm(data_loader)):
        model_input, categories = item
        model_input = model_input.to(device)
        original_label = [my_bidict[item] for item in categories]
        original_label = torch.tensor(original_label, dtype=torch.int64).to(device)
        answer = get_label(model, model_input, device)
        correct_num = torch.sum(answer == original_label)
        acc_tracker.update(correct_num.item(), model_input.shape[0])
    
    return acc_tracker.get_ratio()
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-i', '--data_dir', type=str,
                        default='data', help='Location for the dataset')
    parser.add_argument('-b', '--batch_size', type=int,
                        default=32, help='Batch size for inference')
    parser.add_argument('-m', '--mode', type=str,
                        default='validation', help='Mode for the dataset')
    
    args = parser.parse_args()
    pprint(args.__dict__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {'num_workers':0, 'pin_memory':True, 'drop_last':False}

    ds_transforms = transforms.Compose([transforms.Resize((32, 32)), rescaling])
    dataloader = torch.utils.data.DataLoader(CPEN455Dataset(root_dir=args.data_dir, 
                                                            mode = args.mode, 
                                                            transform=ds_transforms), 
                                             batch_size=args.batch_size, 
                                             shuffle=True, 
                                             **kwargs)

    #Write your code here
    #You should replace the random classifier with your trained model
    #Begin of your code
    #Load your model and evaluate the accuracy on the validation set
    model = PixelCNN(nr_resnet=1, nr_filters=40, input_channels=3, nr_logistic_mix=5)
    # model = PixelCNN(nr_resnet=1, nr_filters=40, input_channels=3, nr_logistic_mix=5)



    #End of your code
    
    model = model.to(device)
    #Attention: the path of the model is fixed to 'models/conditional_pixelcnn.pth'
    #You should save your model to this path
    model.load_state_dict(torch.load('models/conditional_pixelcnn.pth'))
    model.eval()
    print('model parameters loaded')
    acc = classifier(model = model, data_loader = dataloader, device = device)
    print(f"Accuracy: {acc}")
        

    #write a function to get the labels of all the images in the dataset andwrite it to a csv file

    # get the labels of all the images in the dataset

    # write the labels to a csv file

    # get the labels of all the images in the dataset
    # all_labels = []
    # for batch_idx, item in enumerate(tqdm(dataloader)):
    #     model_input, categories = item
    #     model_input = model_input.to(device)
    #     original_label = [my_bidict[item] for item in categories]
    #     original_label = torch.tensor(original_label, dtype=torch.int64).to(device)
    #     answer = get_label(model, model_input, device)
    #     all_labels.append(answer)
    # all_labels = torch.cat(all_labels, dim=0)
    # # write the labels to a csv file
    # import csv
    # with open('labels.csv', mode='w') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(all_labels)

    # print("Labels written to labels.csv")


