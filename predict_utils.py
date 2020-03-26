import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from statistics import mode
from torchvision.transforms import ToTensor

def recover_shape(device, test_path, model):
    img = cv2.imread(test_path)
    background = np.full([img.shape[0],img.shape[1],3], 0.)
    decision = {}
    for i in np.arange(0, img.shape[0] - 8, 1):
        for j in np.arange(0, img.shape[1] - 8, 1):
            img_window = img[i:i+8, j:j+8, :]
            # try transposing instead first and then adding in another dimension
            torch_img = torch.from_numpy(img_window).reshape(1, 3, 8, 8).to(device)
            output = model(torch_img.float())
            _, preds_tensor = torch.max(output, 1)
            prediction = np.squeeze(preds_tensor.cpu().numpy()).item(0)
            print('Prediction is: ', prediction)
            for i_index in range(i, i+8):
                for j_index in range(j, j+8):
                    pixel_loc = '{}-{}'.format(i_index, j_index)
                    if pixel_loc not in decision:
                        decision[pixel_loc] = []
                    decision[pixel_loc].append(prediction)
        if i % 100 == 0:
            print('Finished Remasking Row {}'.format(i))

    print('Encoding Complete!')
    for pixel_loc in decision:
        try:
            majority_vote = mode(decision[pixel_loc])
        except:
            majority_vote = decision[pixel_loc][0]
        print('majority vote', majority_vote)
        i = int(pixel_loc.split('-')[0])
        j = int(pixel_loc.split('-')[1])
        if majority_vote == 0:
            background[i, j, 0] = 0.
            background[i, j, 1] = 0.
            background[i, j, 2] = 255.
        elif majority_vote == 1:
            background[i, j, 0] = 0.
            background[i, j, 1] = 255.
            background[i, j, 2] = 0.
    return background

def test_model(device, model, model_path, test_img_path, fname_path):
    # load back the saved model
    model.load_state_dict(torch.load(model_path))
    # freeze all gradient tracking in forward operation
    for p in model.parameters():
        p.require_grad = False
    # Test result on a simple single-leaf training result
    recovered_mask = recover_shape(device=device, test_path=test_img_path, model=model)
    # show the recovered mask
    plt.imshow(recovered_mask)
    plt.show()

def recover_shape_alt(device, test_path, model, model_path):
    model.load_state_dict(torch.load(model_path))
    img = cv2.imread(test_path)
    mask = np.full((img.shape[0],img.shape[1],3), 0.)
    patch_predictions = {}
    with torch.no_grad():
        for i in np.arange(0, img.shape[0] - 16, 8):
            for j in np.arange(0, img.shape[1] - 16, 8):
                window = np.transpose(img[i:i+16, j:j+16, :], (2, 0, 1)).copy()
                window = torch.from_numpy(window).float().div(255)
                img_tensor = window.unsqueeze(0).to(device)
                output = model(img_tensor)
                _, predicted = torch.max(output, 1)
                patch_predictions[str(i) + '-' + str(j)] = predicted.cpu().numpy().item()
        for patch_coordinate in patch_predictions:
            predicted_class = patch_predictions[patch_coordinate]
            start_indices = patch_coordinate.split('-')
            start_i, start_j = int(start_indices[0]), int(start_indices[1])
            mask_color = [0., 0., 0.]
            if predicted_class == 0:
                mask_color = [0, 255., 0]
            elif predicted_class == 1:
                mask_color = [0, 0, 255.]
            mask[start_i:start_i+16,start_j:start_j+16,:] = np.full((16, 16, 3), np.array(mask_color))
    return mask
            


def test_model_alt(device, model, model_path, test_img_path, fname_path):
    # load back the saved model
    model.load_state_dict(torch.load(model_path))
    # freeze all gradient tracking in forward operation
    for p in model.parameters():
        p.require_grad = False
    # Test result on a simple single-leaf training result
    recovered_mask = recover_shape_alt(device=device, test_path=test_img_path, model=model)
    # show the recovered mask
    plt.imshow(recovered_mask)
    plt.show()

def test_classification_accuracy(device, model, model_path, test_loader):
    model.load_state_dict(torch.load(model_path))
    class_correct = list(0. for i in range(3))
    class_total = list(0. for i in range(3))
    with torch.no_grad():
        for data, target in test_loader:
            images, labels = data.to(device), target.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(c.shape[0]):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    for i in range(3):
        print('Accuracy of Type %d : %2d %%' % (
            i, 100 * class_correct[i] / class_total[i]))

