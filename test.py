#!/home/mirkwood/Miniconda/bin/python
import os
import cnn_face_detect as cfd

curr_dir = os.path.dirname(os.path.realpath(__file__))
train_dataset_path = os.path.join(curr_dir, 'dataset/training.csv')
test_dataset_path = os.path.join(curr_dir, 'dataset/test.csv')
model_output_path = os.path.join(curr_dir, 'model/model_output.h5')


if __name__ == '__main__':

    model = cfd.CNNFaceDetect()
    model.load_training_data(path=train_dataset_path, dataset='montreal')
    model.build_model(savepath=model_output_path)
    model.load_test_data(path=test_dataset_path, dataset='montreal')
    model.predict()

    # cfd.Helper.show_train_image(images=model.Xtrain, labels=model.Ytrain, im_to_show=81, figsize=(12, 12))
    # cfd.Helper.show_test_image(images=model.Xtest, im_to_show=81, figsize=(12, 12))
    cfd.Helper.show_image_wlabels(images=model.Xtest, labels=model.Ytest, im_to_show=25)
